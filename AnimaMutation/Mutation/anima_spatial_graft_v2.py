# coding=utf-8
# Mutation/anima_spatial_graft_v2.py
#
# Anima Spatial Graft V2 / MUDD-Former runtime mutation.
#
# 该文件只包含 ComfyUI 推理所需内容：
#
#   1. MUDD-Former 新增模块；
#   2. MUDD persistent memory 前向逻辑；
#   3. 原版 Anima block/model forward 的原地安装；
#   4. checkpoint / LoRA 架构检测；
#   5. 根据 checkpoint 权重形状推断运行配置；
#   6. 严格的 dtype/device 一致性检查。
#
# 原始 Anima 参数路径保持不变。
#
# 新增参数路径仅位于：
#
#     blocks.{index}.mudd_graft.*
#
# MUTATION_ID 必须与文件名去掉 .py 后完全一致。
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
import re
import weakref
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置
# ============================================================


@dataclass
class MUDDGraftRuntimeConfig:
    """
    MUDD-Former 推理配置。

    MUDD：

        Multi-scale Unified Dynamic Dense-memory Former

    默认值与训练架构 anima_grafted2.py 保持一致。

    hidden override 字段只用于从 checkpoint 权重形状恢复架构，
    不会产生额外参数。
    """

    enabled: bool = True

    # 每隔多少个原始 Transformer block 安装一个 MUDD 节点。
    block_stride: int = 4

    # memory channel 相对于 model_channels 的比例。
    memory_ratio: float = 0.125

    # 如果 checkpoint 可以推断出精确 channel，则优先使用该值。
    memory_channels_override: Optional[int] = None

    # 低分辨率 memory 的空间池化倍数。
    memory_pool: int = 4

    # MUDDMemoryUnit 数量。
    memory_depth: int = 2

    # 是否启用全分辨率 detail 分支。
    use_detail_branch: bool = True

    # detail channel 相对于 model_channels 的比例。
    detail_ratio: float = 0.0625

    # 如果 checkpoint 可以推断出精确 channel，则优先使用该值。
    detail_channels_override: Optional[int] = None

    spatial_kernel_size: int = 3
    temporal_kernel_size: int = 1

    # 推理时通常为 0。
    dropout: float = 0.0

    branch_scale_init: float = 1.0
    memory_layer_scale_init: float = 0.1
    detail_layer_scale_init: float = 0.1

    # sigmoid(-1.5) ~= 0.182。
    memory_update_bias: float = -1.5

    use_framewise_timestep: bool = True

    # 原训练架构默认使用 RMSNorm。
    use_rms_norm: bool = True


# ============================================================
# dtype / device 工具
# ============================================================


def _module_first_floating_parameter(
    module: Optional[nn.Module],
) -> Optional[nn.Parameter]:
    if module is None:
        return None

    try:
        for parameter in module.parameters():
            if (
                torch.is_tensor(parameter)
                and parameter.is_floating_point()
            ):
                return parameter
    except Exception:
        pass

    return None


def _find_model_reference_parameter(
    model: nn.Module,
) -> Optional[nn.Parameter]:
    """
    查找新增 Mutation 模块应当跟随的 dtype/device。

    优先使用 x_embedder，因为 latent 首先进入该模块，它的权重
    通常最能代表 diffusion model 的真实计算精度。
    """

    preferred_modules = (
        getattr(model, "x_embedder", None),
        getattr(model, "t_embedder", None),
        getattr(model, "final_layer", None),
    )

    for module in preferred_modules:
        parameter = _module_first_floating_parameter(
            module
        )

        if parameter is not None:
            return parameter

    try:
        for parameter in model.parameters():
            if (
                torch.is_tensor(parameter)
                and parameter.is_floating_point()
            ):
                return parameter
    except Exception:
        pass

    return None


def _require_same_device(
    tensor: torch.Tensor,
    expected_device: torch.device,
    tensor_name: str,
):
    if tensor.device != expected_device:
        raise RuntimeError(
            f"[SpatialGraftV2/MUDD] {tensor_name} 与新增模块不在"
            "同一设备："
            f"tensor={tensor.device}, module={expected_device}。\n"
            "这通常表示 ComfyUI 模型调度或 block swap 没有把 "
            "mudd_graft 模块移动到正确设备。"
        )


def _cast_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    将 Tensor 严格恢复到新增模块权重的 dtype/device。

    本架构不允许 Norm、interpolate、标量运算或 residual add
    把结果意外泄漏为 FP32。
    """

    if (
        tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = tensor.to(
            device=device,
            dtype=dtype,
        )

    return tensor


def _check_finite(
    tensor: torch.Tensor,
    tensor_name: str,
):
    if not tensor.is_floating_point():
        return

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"[SpatialGraftV2/MUDD] {tensor_name} 中检测到 "
            "NaN/Inf。已终止推理，避免继续生成纯黑图或损坏结果。"
        )


def _validate_odd_kernel(
    kernel_size: int,
    name: str,
):
    kernel_size = int(kernel_size)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"{name} 必须是正奇数，实际为 {kernel_size}"
        )


# ============================================================
# Normalization
# ============================================================


class SimpleRMSNorm(nn.Module):
    """
    与训练架构兼容的无 affine RMSNorm。

    内部使用 FP32 计算 RMS，输出前恢复输入 dtype。
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.dim = int(dim)
        self.eps = float(eps)

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(
                    self.dim,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter(
                "weight",
                None,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        input_device = x.device

        x_float = x.float()

        inverse_rms = torch.rsqrt(
            x_float.square().mean(
                dim=-1,
                keepdim=True,
            ) + self.eps
        )

        output = (
            x_float * inverse_rms
        ).to(
            device=input_device,
            dtype=input_dtype,
        )

        if self.weight is not None:
            output = output * self.weight.to(
                device=input_device,
                dtype=input_dtype,
            )

            output = output.to(
                device=input_device,
                dtype=input_dtype,
            )

        return output


class ChannelNorm3d(nn.Module):
    """
    对 B,C,T,H,W Tensor 的 channel 维执行归一化。

    归一化实际在最后一个维度进行，避免使用空间统计。
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        self.channels = int(channels)

        if use_rms_norm:
            self.norm = SimpleRMSNorm(
                self.channels,
                eps=eps,
                elementwise_affine=False,
            )
        else:
            self.norm = nn.LayerNorm(
                self.channels,
                eps=eps,
                elementwise_affine=False,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        input_device = x.device

        x = x.permute(
            0,
            2,
            3,
            4,
            1,
        )

        x = self.norm(x)

        x = x.to(
            device=input_device,
            dtype=input_dtype,
        )

        return x.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()


# ============================================================
# MUDD 低成本处理单元
# ============================================================


class MUDDMemoryUnit(nn.Module):
    """
    低分辨率时空 memory residual unit。

    参数路径与训练架构保持一致：

        memory_units.{index}.depthwise.weight
        memory_units.{index}.pointwise_in.weight
        memory_units.{index}.pointwise_out.weight
        memory_units.{index}.layer_scale
    """

    def __init__(
        self,
        channels: int,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 1,
        dropout: float = 0.0,
        layer_scale_init: float = 0.1,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        channels = int(channels)

        _validate_odd_kernel(
            spatial_kernel_size,
            "spatial_kernel_size",
        )

        _validate_odd_kernel(
            temporal_kernel_size,
            "temporal_kernel_size",
        )

        if layer_scale_init <= 0.0:
            raise ValueError(
                "memory_layer_scale_init 必须大于 0"
            )

        self.channels = channels

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(
                int(temporal_kernel_size),
                int(spatial_kernel_size),
                int(spatial_kernel_size),
            ),
            stride=1,
            padding=(
                int(temporal_kernel_size) // 2,
                int(spatial_kernel_size) // 2,
                int(spatial_kernel_size) // 2,
            ),
            groups=channels,
            bias=False,
        )

        self.norm = ChannelNorm3d(
            channels,
            eps=1e-6,
            use_rms_norm=use_rms_norm,
        )

        self.pointwise_in = nn.Conv3d(
            channels,
            channels * 2,
            kernel_size=1,
            bias=False,
        )

        self.activation = nn.GELU(
            approximate="tanh"
        )

        self.pointwise_out = nn.Conv3d(
            channels * 2,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = (
            nn.Dropout3d(float(dropout))
            if dropout > 0.0
            else nn.Identity()
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                float(layer_scale_init),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = self.depthwise.weight.dtype
        compute_device = self.depthwise.weight.device

        _require_same_device(
            x,
            compute_device,
            "MUDDMemoryUnit 输入",
        )

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        residual = x

        branch = self.depthwise(x)
        branch = self.norm(branch)

        branch = _cast_tensor(
            branch,
            dtype=compute_dtype,
            device=compute_device,
        )

        branch = self.pointwise_in(branch)
        branch = self.activation(branch)
        branch = self.pointwise_out(branch)
        branch = self.dropout(branch)

        branch = _cast_tensor(
            branch,
            dtype=compute_dtype,
            device=compute_device,
        )

        scale = self.layer_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
            1,
        )

        output = residual + scale * branch

        return _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )


class MUDDDetailUnit(nn.Module):
    """
    全分辨率局部细节分支。

    参数路径与训练架构保持一致：

        detail_unit.depthwise.weight
        detail_unit.pointwise.weight
        detail_unit.layer_scale
    """

    def __init__(
        self,
        channels: int,
        spatial_kernel_size: int = 3,
        dropout: float = 0.0,
        layer_scale_init: float = 0.1,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        channels = int(channels)

        _validate_odd_kernel(
            spatial_kernel_size,
            "spatial_kernel_size",
        )

        if layer_scale_init <= 0.0:
            raise ValueError(
                "detail_layer_scale_init 必须大于 0"
            )

        self.channels = channels

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(
                1,
                int(spatial_kernel_size),
                int(spatial_kernel_size),
            ),
            stride=1,
            padding=(
                0,
                int(spatial_kernel_size) // 2,
                int(spatial_kernel_size) // 2,
            ),
            groups=channels,
            bias=False,
        )

        self.norm = ChannelNorm3d(
            channels,
            eps=1e-6,
            use_rms_norm=use_rms_norm,
        )

        self.pointwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.activation = nn.GELU(
            approximate="tanh"
        )

        self.dropout = (
            nn.Dropout3d(float(dropout))
            if dropout > 0.0
            else nn.Identity()
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                float(layer_scale_init),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = self.depthwise.weight.dtype
        compute_device = self.depthwise.weight.device

        _require_same_device(
            x,
            compute_device,
            "MUDDDetailUnit 输入",
        )

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        residual = x

        branch = self.depthwise(x)
        branch = self.norm(branch)

        branch = _cast_tensor(
            branch,
            dtype=compute_dtype,
            device=compute_device,
        )

        branch = self.pointwise(branch)
        branch = self.activation(branch)
        branch = self.dropout(branch)

        branch = _cast_tensor(
            branch,
            dtype=compute_dtype,
            device=compute_device,
        )

        scale = self.layer_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
            1,
        )

        output = residual + scale * branch

        return _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )


# ============================================================
# MUDD-Former Graft
# ============================================================


class MUDDFormerGraft(nn.Module):
    """
    Anima MUDD-Former runtime graft。

    输入：

        x_B_T_H_W_D:
            当前深层 Transformer 特征。

        shallow_B_T_H_W_D:
            第一个 Transformer block 之前的浅层特征。

        previous_memory:
            上一个 MUDD 节点产生的低分辨率 persistent memory。

        timestep_embedding_B_T_D:
            Anima timestep embedding。

    输出：

        output:
            注入 MUDD residual 后的 Transformer 特征。

        updated_memory:
            传给下一个 MUDD 节点的 persistent memory。
    """

    def __init__(
        self,
        model_channels: int,
        config: MUDDGraftRuntimeConfig,
        block_index: int,
    ):
        super().__init__()

        self.model_channels = int(
            model_channels
        )

        self.block_index = int(
            block_index
        )

        self.config = copy.deepcopy(
            config
        )

        if (
            config.memory_channels_override
            is not None
        ):
            memory_channels = int(
                config.memory_channels_override
            )

            if memory_channels <= 0:
                raise ValueError(
                    "memory_channels_override 必须大于 0"
                )
        else:
            memory_channels = max(
                32,
                int(
                    round(
                        self.model_channels
                        * float(config.memory_ratio)
                    )
                ),
            )

            memory_channels = int(
                math.ceil(
                    memory_channels / 32
                ) * 32
            )

            memory_channels = min(
                memory_channels,
                self.model_channels,
            )

        if (
            config.detail_channels_override
            is not None
        ):
            detail_channels = int(
                config.detail_channels_override
            )

            if detail_channels <= 0:
                raise ValueError(
                    "detail_channels_override 必须大于 0"
                )
        else:
            detail_channels = max(
                32,
                int(
                    round(
                        self.model_channels
                        * float(config.detail_ratio)
                    )
                ),
            )

            detail_channels = int(
                math.ceil(
                    detail_channels / 32
                ) * 32
            )

            detail_channels = min(
                detail_channels,
                self.model_channels,
            )

        self.memory_channels = int(
            memory_channels
        )

        self.detail_channels = int(
            detail_channels
        )

        if int(config.memory_depth) < 0:
            raise ValueError(
                "memory_depth 不能小于 0"
            )

        if int(config.memory_pool) <= 0:
            raise ValueError(
                "memory_pool 必须大于 0"
            )

        _validate_odd_kernel(
            config.spatial_kernel_size,
            "spatial_kernel_size",
        )

        _validate_odd_kernel(
            config.temporal_kernel_size,
            "temporal_kernel_size",
        )

        if config.use_rms_norm:
            self.input_norm = SimpleRMSNorm(
                self.model_channels,
                eps=1e-6,
                elementwise_affine=False,
            )

            self.shallow_norm = SimpleRMSNorm(
                self.model_channels,
                eps=1e-6,
                elementwise_affine=False,
            )
        else:
            self.input_norm = nn.LayerNorm(
                self.model_channels,
                eps=1e-6,
                elementwise_affine=False,
            )

            self.shallow_norm = nn.LayerNorm(
                self.model_channels,
                eps=1e-6,
                elementwise_affine=False,
            )

        # 当前深层特征 -> memory space。
        self.current_memory_proj = nn.Conv3d(
            self.model_channels,
            self.memory_channels,
            kernel_size=1,
            bias=False,
        )

        # 初始浅层特征 -> memory space。
        self.shallow_memory_proj = nn.Conv3d(
            self.model_channels,
            self.memory_channels,
            kernel_size=1,
            bias=False,
        )

        # current + shallow + previous memory。
        self.memory_merge = nn.Conv3d(
            self.memory_channels * 3,
            self.memory_channels,
            kernel_size=1,
            bias=False,
        )

        self.memory_units = nn.ModuleList(
            [
                MUDDMemoryUnit(
                    channels=self.memory_channels,
                    spatial_kernel_size=(
                        config.spatial_kernel_size
                    ),
                    temporal_kernel_size=(
                        config.temporal_kernel_size
                    ),
                    dropout=config.dropout,
                    layer_scale_init=(
                        config.memory_layer_scale_init
                    ),
                    use_rms_norm=(
                        config.use_rms_norm
                    ),
                )
                for _ in range(
                    int(config.memory_depth)
                )
            ]
        )

        # 输出三个 memory-width 调制值：
        #
        #   1. update logits；
        #   2. output gate；
        #   3. shallow gate。
        self.time_modulation = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.model_channels,
                self.memory_channels,
                bias=False,
            ),

            nn.SiLU(),

            nn.Linear(
                self.memory_channels,
                3 * self.memory_channels,
                bias=True,
            ),
        )

        if config.use_detail_branch:
            self.detail_in_proj = nn.Conv3d(
                self.model_channels,
                self.detail_channels,
                kernel_size=1,
                bias=False,
            )

            self.detail_unit = MUDDDetailUnit(
                channels=self.detail_channels,
                spatial_kernel_size=(
                    config.spatial_kernel_size
                ),
                dropout=config.dropout,
                layer_scale_init=(
                    config.detail_layer_scale_init
                ),
                use_rms_norm=(
                    config.use_rms_norm
                ),
            )

            self.detail_to_memory = nn.Conv3d(
                self.detail_channels,
                self.memory_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.detail_in_proj = None
            self.detail_unit = None
            self.detail_to_memory = None

        self.output_proj = nn.Conv3d(
            self.memory_channels,
            self.model_channels,
            kernel_size=1,
            bias=False,
        )

        self.branch_scale = nn.Parameter(
            torch.tensor(
                float(config.branch_scale_init),
                dtype=torch.float32,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.current_memory_proj.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.shallow_memory_proj.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.memory_merge.weight,
            a=math.sqrt(5),
        )

        for unit in self.memory_units:
            nn.init.kaiming_uniform_(
                unit.depthwise.weight,
                a=math.sqrt(5),
            )

            nn.init.kaiming_uniform_(
                unit.pointwise_in.weight,
                a=math.sqrt(5),
            )

            nn.init.kaiming_uniform_(
                unit.pointwise_out.weight,
                a=math.sqrt(5),
            )

        if self.detail_in_proj is not None:
            nn.init.kaiming_uniform_(
                self.detail_in_proj.weight,
                a=math.sqrt(5),
            )

            nn.init.kaiming_uniform_(
                self.detail_unit.depthwise.weight,
                a=math.sqrt(5),
            )

            nn.init.kaiming_uniform_(
                self.detail_unit.pointwise.weight,
                a=math.sqrt(5),
            )

            nn.init.kaiming_uniform_(
                self.detail_to_memory.weight,
                a=math.sqrt(5),
            )

        nn.init.normal_(
            self.time_modulation[1].weight,
            mean=0.0,
            std=(
                1.0
                / math.sqrt(
                    self.model_channels
                )
            ),
        )

        nn.init.zeros_(
            self.time_modulation[3].weight
        )

        nn.init.zeros_(
            self.time_modulation[3].bias
        )

        with torch.no_grad():
            memory_channels = (
                self.memory_channels
            )

            # memory update gate。
            self.time_modulation[3].bias[
                :memory_channels
            ].fill_(
                float(
                    self.config.memory_update_bias
                )
            )

            # output gate。
            self.time_modulation[3].bias[
                memory_channels:
                2 * memory_channels
            ].zero_()

            # shallow memory gate。
            self.time_modulation[3].bias[
                2 * memory_channels:
            ].zero_()

        # 未加载 Mutation 权重时保持原模型函数不变。
        nn.init.zeros_(
            self.output_proj.weight
        )

    def _pool_memory(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        pool = int(
            self.config.memory_pool
        )

        if pool <= 1:
            return x

        input_dtype = x.dtype
        input_device = x.device

        x = F.avg_pool3d(
            x,
            kernel_size=(
                1,
                pool,
                pool,
            ),
            stride=(
                1,
                pool,
                pool,
            ),
            ceil_mode=True,
            count_include_pad=False,
        )

        return _cast_tensor(
            x,
            dtype=input_dtype,
            device=input_device,
        )

    def _prepare_timestep(
        self,
        timestep_embedding: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(
            timestep_embedding
        ):
            raise TypeError(
                "MUDD timestep embedding 必须是 Tensor"
            )

        if timestep_embedding.ndim != 3:
            raise ValueError(
                "MUDD timestep embedding 期望 [B,T,D]，"
                f"实际为 {tuple(timestep_embedding.shape)}"
            )

        if (
            timestep_embedding.shape[0]
            != batch_size
        ):
            raise ValueError(
                "MUDD timestep batch 与特征 batch 不一致："
                f"{timestep_embedding.shape[0]} != "
                f"{batch_size}"
            )

        if (
            timestep_embedding.shape[-1]
            != self.model_channels
        ):
            raise ValueError(
                "MUDD timestep channel 与 model_channels "
                "不一致："
                f"{timestep_embedding.shape[-1]} != "
                f"{self.model_channels}"
            )

        timestep_frames = int(
            timestep_embedding.shape[1]
        )

        if self.config.use_framewise_timestep:
            if (
                timestep_frames == 1
                and num_frames > 1
            ):
                timestep_embedding = (
                    timestep_embedding.expand(
                        batch_size,
                        num_frames,
                        self.model_channels,
                    )
                )
            elif timestep_frames != num_frames:
                raise ValueError(
                    "MUDD timestep 帧数与特征帧数不一致："
                    f"{timestep_frames} != {num_frames}"
                )
        else:
            timestep_embedding = (
                timestep_embedding[
                    :,
                    :1,
                    :,
                ].expand(
                    batch_size,
                    num_frames,
                    self.model_channels,
                )
            )

        return timestep_embedding.contiguous()

    def _run_memory_units(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for unit in self.memory_units:
            x = unit(x)

        return x

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        shallow_B_T_H_W_D: torch.Tensor,
        previous_memory: Optional[torch.Tensor],
        timestep_embedding_B_T_D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(
            x_B_T_H_W_D
        ):
            raise TypeError(
                "MUDD 当前特征必须是 Tensor"
            )

        if not torch.is_tensor(
            shallow_B_T_H_W_D
        ):
            raise TypeError(
                "MUDD shallow feature 必须是 Tensor"
            )

        if x_B_T_H_W_D.ndim != 5:
            raise ValueError(
                "MUDD 当前特征期望 [B,T,H,W,D]，"
                f"实际为 {tuple(x_B_T_H_W_D.shape)}"
            )

        if shallow_B_T_H_W_D.ndim != 5:
            raise ValueError(
                "MUDD shallow feature 期望 [B,T,H,W,D]，"
                f"实际为 {tuple(shallow_B_T_H_W_D.shape)}"
            )

        if (
            shallow_B_T_H_W_D.shape
            != x_B_T_H_W_D.shape
        ):
            raise ValueError(
                "MUDD shallow feature 与当前特征形状不一致："
                f"{tuple(shallow_B_T_H_W_D.shape)} != "
                f"{tuple(x_B_T_H_W_D.shape)}"
            )

        compute_dtype = (
            self.current_memory_proj.weight.dtype
        )

        compute_device = (
            self.current_memory_proj.weight.device
        )

        _require_same_device(
            x_B_T_H_W_D,
            compute_device,
            "MUDD 当前特征",
        )

        _require_same_device(
            shallow_B_T_H_W_D,
            compute_device,
            "MUDD shallow feature",
        )

        _require_same_device(
            timestep_embedding_B_T_D,
            compute_device,
            "MUDD timestep embedding",
        )

        # 所有 residual 和输入严格对齐到 graft 权重 dtype。
        x_B_T_H_W_D = _cast_tensor(
            x_B_T_H_W_D,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_B_T_H_W_D = _cast_tensor(
            shallow_B_T_H_W_D,
            dtype=compute_dtype,
            device=compute_device,
        )

        timestep_embedding_B_T_D = _cast_tensor(
            timestep_embedding_B_T_D,
            dtype=compute_dtype,
            device=compute_device,
        )

        residual = x_B_T_H_W_D

        (
            batch_size,
            num_frames,
            height,
            width,
            channels,
        ) = x_B_T_H_W_D.shape

        if channels != self.model_channels:
            raise ValueError(
                "MUDD 输入 channel 不匹配："
                f"input={channels}, "
                f"expected={self.model_channels}"
            )

        timestep_embedding_B_T_D = (
            self._prepare_timestep(
                timestep_embedding_B_T_D,
                batch_size=batch_size,
                num_frames=num_frames,
            )
        )

        current_norm = self.input_norm(
            x_B_T_H_W_D
        )

        shallow_norm = self.shallow_norm(
            shallow_B_T_H_W_D
        )

        current_norm = _cast_tensor(
            current_norm,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_norm = _cast_tensor(
            shallow_norm,
            dtype=compute_dtype,
            device=compute_device,
        )

        current_3d = current_norm.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        shallow_3d = shallow_norm.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        current_memory = (
            self.current_memory_proj(
                current_3d
            )
        )

        shallow_memory = (
            self.shallow_memory_proj(
                shallow_3d
            )
        )

        current_memory = _cast_tensor(
            current_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_memory = _cast_tensor(
            shallow_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        current_memory = self._pool_memory(
            current_memory
        )

        shallow_memory = self._pool_memory(
            shallow_memory
        )

        if previous_memory is None:
            previous_memory = shallow_memory
        else:
            if not torch.is_tensor(
                previous_memory
            ):
                raise TypeError(
                    "MUDD previous_memory 必须是 Tensor 或 None"
                )

            if previous_memory.ndim != 5:
                raise ValueError(
                    "MUDD previous_memory 期望 "
                    "[B,C,T,H,W]，实际为 "
                    f"{tuple(previous_memory.shape)}"
                )

            if (
                previous_memory.shape[0]
                != batch_size
            ):
                raise ValueError(
                    "MUDD previous_memory batch 不一致："
                    f"{previous_memory.shape[0]} != "
                    f"{batch_size}"
                )

            if (
                previous_memory.shape[1]
                != self.memory_channels
            ):
                raise ValueError(
                    "MUDD previous_memory channel 不一致："
                    f"{previous_memory.shape[1]} != "
                    f"{self.memory_channels}"
                )

            previous_memory = _cast_tensor(
                previous_memory,
                dtype=compute_dtype,
                device=compute_device,
            )

            if (
                previous_memory.shape[-3:]
                != current_memory.shape[-3:]
            ):
                previous_memory = F.interpolate(
                    previous_memory,
                    size=current_memory.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

                previous_memory = _cast_tensor(
                    previous_memory,
                    dtype=compute_dtype,
                    device=compute_device,
                )

        time_parameter = (
            _module_first_floating_parameter(
                self.time_modulation
            )
        )

        if time_parameter is not None:
            timestep_embedding_B_T_D = (
                _cast_tensor(
                    timestep_embedding_B_T_D,
                    dtype=time_parameter.dtype,
                    device=time_parameter.device,
                )
            )

        modulation = self.time_modulation(
            timestep_embedding_B_T_D
        )

        modulation = _cast_tensor(
            modulation,
            dtype=compute_dtype,
            device=compute_device,
        )

        (
            update_logits,
            output_gate,
            shallow_gate,
        ) = modulation.chunk(
            3,
            dim=-1,
        )

        # B,T,C -> B,C,T,1,1
        update_logits = update_logits.permute(
            0,
            2,
            1,
        ).contiguous()[
            :,
            :,
            :,
            None,
            None,
        ]

        output_gate = output_gate.permute(
            0,
            2,
            1,
        ).contiguous()[
            :,
            :,
            :,
            None,
            None,
        ]

        shallow_gate = shallow_gate.permute(
            0,
            2,
            1,
        ).contiguous()[
            :,
            :,
            :,
            None,
            None,
        ]

        update_logits = _cast_tensor(
            update_logits,
            dtype=compute_dtype,
            device=compute_device,
        )

        output_gate = _cast_tensor(
            output_gate,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_gate = _cast_tensor(
            shallow_gate,
            dtype=compute_dtype,
            device=compute_device,
        )

        memory_update_strength = torch.sigmoid(
            update_logits
        )

        one = torch.ones(
            (),
            device=compute_device,
            dtype=compute_dtype,
        )

        half = torch.tensor(
            0.5,
            device=compute_device,
            dtype=compute_dtype,
        )

        output_strength = (
            one
            + half * torch.tanh(
                output_gate
            )
        )

        shallow_strength = (
            one
            + half * torch.tanh(
                shallow_gate
            )
        )

        memory_update_strength = _cast_tensor(
            memory_update_strength,
            dtype=compute_dtype,
            device=compute_device,
        )

        output_strength = _cast_tensor(
            output_strength,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_strength = _cast_tensor(
            shallow_strength,
            dtype=compute_dtype,
            device=compute_device,
        )

        merged_memory = torch.cat(
            [
                current_memory,
                shallow_memory
                * shallow_strength,
                previous_memory,
            ],
            dim=1,
        )

        merged_memory = _cast_tensor(
            merged_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        memory_candidate = self.memory_merge(
            merged_memory
        )

        memory_candidate = _cast_tensor(
            memory_candidate,
            dtype=compute_dtype,
            device=compute_device,
        )

        memory_candidate = (
            self._run_memory_units(
                memory_candidate
            )
        )

        updated_memory = (
            previous_memory
            + memory_update_strength
            * (
                memory_candidate
                - previous_memory
            )
        )

        updated_memory = _cast_tensor(
            updated_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        restored_memory = F.interpolate(
            updated_memory,
            size=(
                num_frames,
                height,
                width,
            ),
            mode="trilinear",
            align_corners=False,
        )

        restored_memory = _cast_tensor(
            restored_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        if self.detail_in_proj is not None:
            detail = self.detail_in_proj(
                current_3d
            )

            detail = _cast_tensor(
                detail,
                dtype=compute_dtype,
                device=compute_device,
            )

            detail = self.detail_unit(
                detail
            )

            detail = self.detail_to_memory(
                detail
            )

            detail = _cast_tensor(
                detail,
                dtype=compute_dtype,
                device=compute_device,
            )

            restored_memory = (
                restored_memory + detail
            )

            restored_memory = _cast_tensor(
                restored_memory,
                dtype=compute_dtype,
                device=compute_device,
            )

        output_feature = (
            restored_memory
            * output_strength
        )

        output_feature = _cast_tensor(
            output_feature,
            dtype=compute_dtype,
            device=compute_device,
        )

        output_feature = self.output_proj(
            output_feature
        )

        output_feature = _cast_tensor(
            output_feature,
            dtype=compute_dtype,
            device=compute_device,
        )

        output_feature = output_feature.permute(
            0,
            2,
            3,
            4,
            1,
        ).contiguous()

        branch_scale = self.branch_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        )

        output = (
            residual
            + branch_scale
            * output_feature
        )

        # 最终输出严格保持 graft 权重 dtype。
        output = _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )

        updated_memory = _cast_tensor(
            updated_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        _check_finite(
            output,
            "MUDD 输出",
        )

        _check_finite(
            updated_memory,
            "MUDD updated_memory",
        )

        return output, updated_memory


# ============================================================
# 默认索引生成
# ============================================================


def build_every_n_block_indices(
    num_blocks: int,
    stride: int = 4,
    include_last_partial: bool = False,
) -> List[int]:
    """
    stride=4 时返回：

        [3, 7, 11, 15, ...]

    表示每完成四个原始 Transformer blocks 后运行一次 MUDD。
    """

    num_blocks = int(num_blocks)
    stride = int(stride)

    if num_blocks <= 0:
        return []

    if stride <= 0:
        raise ValueError(
            f"stride 必须大于 0，实际为 {stride}"
        )

    indices = list(
        range(
            stride - 1,
            num_blocks,
            stride,
        )
    )

    if (
        include_last_partial
        and num_blocks - 1 not in indices
    ):
        indices.append(
            num_blocks - 1
        )

    return sorted(
        set(indices)
    )


# ============================================================
# Runtime context / Forward 安装
# ============================================================


def _get_or_create_runtime_context(
    model: nn.Module,
) -> Dict[str, Optional[torch.Tensor]]:
    context = getattr(
        model,
        "_mudd_runtime_context",
        None,
    )

    if context is None:
        context = {
            "shallow": None,
            "memory": None,
        }

        object.__setattr__(
            model,
            "_mudd_runtime_context",
            context,
        )

    return context


def _extract_primary_tensor(
    output,
) -> Tuple[torch.Tensor, Optional[str]]:
    """
    正常 Anima block 直接返回 Tensor。

    为兼容少量修改版运行时，也支持 tuple/list 的第一个元素为
    hidden_states Tensor。
    """

    if torch.is_tensor(output):
        return output, None

    if (
        isinstance(output, tuple)
        and len(output) > 0
        and torch.is_tensor(output[0])
    ):
        return output[0], "tuple"

    if (
        isinstance(output, list)
        and len(output) > 0
        and torch.is_tensor(output[0])
    ):
        return output[0], "list"

    raise TypeError(
        "[SpatialGraftV2/MUDD] 原始 Anima block.forward "
        f"返回了不支持的类型：{type(output).__name__}。\n"
        "期望 Tensor，或第一个元素为 Tensor 的 tuple/list。"
    )


def _replace_primary_tensor(
    original_output,
    new_tensor: torch.Tensor,
    container_type: Optional[str],
):
    if container_type is None:
        return new_tensor

    if container_type == "tuple":
        return (
            new_tensor,
            *original_output[1:],
        )

    if container_type == "list":
        result = list(
            original_output
        )

        result[0] = new_tensor

        return result

    raise RuntimeError(
        f"未知输出容器类型：{container_type}"
    )


def _install_mudd_block_forward(
    block: nn.Module,
    model: nn.Module,
    block_index: int,
    apply_mudd: bool,
):
    """
    原地修改 block.forward。

    Block 0 负责记录原始 patch-embedded shallow feature。
    被选中的 blocks 在原始 forward 后运行 mudd_graft。
    """

    if getattr(
        block,
        "_mudd_graft_forward_installed",
        False,
    ):
        existing_apply = bool(
            getattr(
                block,
                "_mudd_graft_apply_after_block",
                False,
            )
        )

        if apply_mudd and not existing_apply:
            object.__setattr__(
                block,
                "_mudd_graft_apply_after_block",
                True,
            )

        return

    if apply_mudd and not hasattr(
        block,
        "mudd_graft",
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 mudd_graft，"
            "不能安装 MUDD forward。"
        )

    original_forward = block.forward

    object.__setattr__(
        block,
        "_original_anima_forward",
        original_forward,
    )

    object.__setattr__(
        block,
        "_mudd_owner_model_ref",
        weakref.ref(model),
    )

    object.__setattr__(
        block,
        "_mudd_runtime_block_index",
        int(block_index),
    )

    object.__setattr__(
        block,
        "_mudd_graft_apply_after_block",
        bool(apply_mudd),
    )

    def mudd_grafted_forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        *args,
        **kwargs,
    ):
        owner_reference = getattr(
            self,
            "_mudd_owner_model_ref",
            None,
        )

        owner_model = (
            owner_reference()
            if owner_reference is not None
            else None
        )

        if owner_model is None:
            raise RuntimeError(
                "[SpatialGraftV2/MUDD] 无法取得所属 Anima "
                "model，runtime context 已失效。"
            )

        context = _get_or_create_runtime_context(
            owner_model
        )

        runtime_block_index = int(
            getattr(
                self,
                "_mudd_runtime_block_index",
                -1,
            )
        )

        # 每次完整前向进入 block 0 时，重置 persistent memory。
        if runtime_block_index == 0:
            if not torch.is_tensor(
                x_B_T_H_W_D
            ):
                raise TypeError(
                    "[SpatialGraftV2/MUDD] Block 0 输入不是 Tensor"
                )

            context["shallow"] = (
                x_B_T_H_W_D
            )

            context["memory"] = None

        original_output = (
            self._original_anima_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                *args,
                **kwargs,
            )
        )

        apply_after_block = bool(
            getattr(
                self,
                "_mudd_graft_apply_after_block",
                False,
            )
        )

        if not apply_after_block:
            return original_output

        hidden_states, container_type = (
            _extract_primary_tensor(
                original_output
            )
        )

        shallow_feature = context.get(
            "shallow"
        )

        if shallow_feature is None:
            raise RuntimeError(
                "[SpatialGraftV2/MUDD] 未捕获 shallow feature。\n"
                "Anima blocks 可能没有从 block 0 顺序执行，或当前 "
                "ComfyUI 版本绕过了标准 Anima forward。"
            )

        hidden_states, updated_memory = (
            self.mudd_graft(
                hidden_states,
                shallow_feature,
                context.get("memory"),
                emb_B_T_D,
            )
        )

        context["memory"] = (
            updated_memory
        )

        return _replace_primary_tensor(
            original_output,
            hidden_states,
            container_type,
        )

    block.forward = MethodType(
        mudd_grafted_forward,
        block,
    )

    object.__setattr__(
        block,
        "_mudd_graft_forward_installed",
        True,
    )


def _install_model_runtime_context_forward(
    model: nn.Module,
):
    """
    包装 forward_mini_train_dit，只负责创建和清理 MUDD context。

    原始 Anima 顶层执行逻辑不被复制，因而能够兼容 ComfyUI 中
    不同版本的 Anima attention 参数、LLM adapter 和 block swap。
    """

    if getattr(
        model,
        "_mudd_model_forward_installed",
        False,
    ):
        return

    if hasattr(
        model,
        "forward_mini_train_dit",
    ):
        forward_name = (
            "forward_mini_train_dit"
        )
    elif hasattr(model, "forward"):
        # 理论上的兼容 fallback。
        forward_name = "forward"
    else:
        raise AttributeError(
            "Anima model 不存在 forward_mini_train_dit "
            "或 forward。"
        )

    original_forward = getattr(
        model,
        forward_name,
    )

    object.__setattr__(
        model,
        "_original_anima_mudd_model_forward",
        original_forward,
    )

    object.__setattr__(
        model,
        "_mudd_wrapped_forward_name",
        forward_name,
    )

    sentinel = object()

    def mudd_model_forward(
        self,
        *args,
        **kwargs,
    ):
        previous_context = getattr(
            self,
            "_mudd_runtime_context",
            sentinel,
        )

        object.__setattr__(
            self,
            "_mudd_runtime_context",
            {
                "shallow": None,
                "memory": None,
            },
        )

        try:
            return (
                self._original_anima_mudd_model_forward(
                    *args,
                    **kwargs,
                )
            )
        finally:
            if previous_context is sentinel:
                try:
                    object.__delattr__(
                        self,
                        "_mudd_runtime_context",
                    )
                except AttributeError:
                    pass
            else:
                object.__setattr__(
                    self,
                    "_mudd_runtime_context",
                    previous_context,
                )

    setattr(
        model,
        forward_name,
        MethodType(
            mudd_model_forward,
            model,
        ),
    )

    object.__setattr__(
        model,
        "_mudd_model_forward_installed",
        True,
    )


# ============================================================
# Mutation 主类
# ============================================================


class GraftedAnima:
    """
    AnimaBaker Mutation 系统入口。

    注意：

    该类不是新的 nn.Module 模型，而是运行时架构识别器和安装器。
    """

    MUTATION_API_VERSION = 1

    MUTATION_ID = (
        "anima_spatial_graft_v2"
    )

    DISPLAY_NAME = (
        "Anima MUDD Spatial Memory Graft V2"
    )

    # 必须保持为 mudd_graft，才能加载训练架构中的参数：
    #
    #     blocks.{index}.mudd_graft.*
    MODULE_NAMESPACE = "mudd_graft"

    DEFAULT_CONFIG = MUDDGraftRuntimeConfig(
        enabled=True,
        block_stride=4,
        memory_ratio=0.125,
        memory_channels_override=None,
        memory_pool=4,
        memory_depth=2,
        use_detail_branch=True,
        detail_ratio=0.0625,
        detail_channels_override=None,
        spatial_kernel_size=3,
        temporal_kernel_size=1,
        dropout=0.0,
        branch_scale_init=1.0,
        memory_layer_scale_init=0.1,
        detail_layer_scale_init=0.1,
        memory_update_bias=-1.5,
        use_framewise_timestep=True,
        use_rms_norm=True,
    )

    @classmethod
    def detect(
        cls,
        state_dict_keys,
    ) -> int:
        """
        根据基础模型或 LoRA 参数键判断是否属于本架构。
        """

        score = 0

        for key in state_dict_keys or []:
            key_lower = str(
                key
            ).lower()

            if cls.MUTATION_ID in key_lower:
                score = max(
                    score,
                    120,
                )

            elif ".mudd_graft." in key_lower:
                score = max(
                    score,
                    105,
                )

            elif key_lower.startswith(
                "mudd_graft."
            ):
                score = max(
                    score,
                    102,
                )

            elif "_mudd_graft_" in key_lower:
                score = max(
                    score,
                    98,
                )

            elif "mudd_graft." in key_lower:
                score = max(
                    score,
                    96,
                )

            elif "mudd_graft_" in key_lower:
                score = max(
                    score,
                    92,
                )

            # 使用多个专有子模块组合进一步识别。
            elif (
                "current_memory_proj" in key_lower
                and "memory_merge" in key_lower
            ):
                score = max(
                    score,
                    75,
                )

        return score

    @classmethod
    def is_mutation_key(
        cls,
        key,
    ) -> bool:
        """
        判断某个参数是否属于 V2 新增层。
        """

        key_lower = str(
            key
        ).lower()

        return (
            ".mudd_graft." in key_lower
            or key_lower.startswith(
                "mudd_graft."
            )
            or "_mudd_graft_" in key_lower
            or "mudd_graft." in key_lower
            or "mudd_graft_" in key_lower
            or cls.MUTATION_ID in key_lower
        )

    @classmethod
    def _infer_indices_from_keys(
        cls,
        source_keys,
    ) -> List[int]:
        indices = set()

        patterns = (
            # 原生 PyTorch：
            #
            # diffusion_model.blocks.3.mudd_graft.output_proj.weight
            re.compile(
                r"(?:^|\.)blocks\.(\d+)"
                r"\.mudd_graft(?:\.|$)",
                re.IGNORECASE,
            ),

            # 常见 LoRA 下划线路径：
            #
            # lora_unet_blocks_3_mudd_graft_output_proj
            re.compile(
                r"(?:^|_)blocks_(\d+)"
                r"_mudd_graft(?:_|$)",
                re.IGNORECASE,
            ),
        )

        for key in source_keys or []:
            key_string = str(key)

            for pattern in patterns:
                match = pattern.search(
                    key_string
                )

                if match is not None:
                    indices.add(
                        int(
                            match.group(1)
                        )
                    )

                    break

        return sorted(indices)

    @classmethod
    def _infer_config_from_state_dict(
        cls,
        source_state_dict: Optional[
            Dict[str, torch.Tensor]
        ],
        model_channels: int,
    ) -> MUDDGraftRuntimeConfig:
        """
        从完整 Mutation checkpoint 的权重形状恢复架构。

        无法由权重推断的 memory_pool、Norm 类型等字段保持默认值。
        """

        config = copy.deepcopy(
            cls.DEFAULT_CONFIG
        )

        if not source_state_dict:
            return config

        model_channels = int(
            model_channels
        )

        memory_channels = set()
        detail_channels = set()
        memory_unit_indices = set()
        spatial_kernel_sizes = set()
        temporal_kernel_sizes = set()

        saw_core_base_weight = False
        saw_detail_base_weight = False
        saw_any_memory_unit_weight = False

        current_memory_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.mudd_graft"
            r"\.current_memory_proj\.weight$",
            re.IGNORECASE,
        )

        shallow_memory_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.mudd_graft"
            r"\.shallow_memory_proj\.weight$",
            re.IGNORECASE,
        )

        memory_merge_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.mudd_graft"
            r"\.memory_merge\.weight$",
            re.IGNORECASE,
        )

        output_proj_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.mudd_graft"
            r"\.output_proj\.weight$",
            re.IGNORECASE,
        )

        memory_unit_pattern = re.compile(
            r"\.mudd_graft\.memory_units\."
            r"(\d+)\.",
            re.IGNORECASE,
        )

        memory_depthwise_pattern = re.compile(
            r"\.mudd_graft\.memory_units\."
            r"(\d+)\.depthwise\.weight$",
            re.IGNORECASE,
        )

        detail_in_pattern = re.compile(
            r"\.mudd_graft"
            r"\.detail_in_proj\.weight$",
            re.IGNORECASE,
        )

        detail_depthwise_pattern = re.compile(
            r"\.mudd_graft"
            r"\.detail_unit\.depthwise\.weight$",
            re.IGNORECASE,
        )

        detail_to_memory_pattern = re.compile(
            r"\.mudd_graft"
            r"\.detail_to_memory\.weight$",
            re.IGNORECASE,
        )

        for key, tensor in (
            source_state_dict.items()
        ):
            if not torch.is_tensor(tensor):
                continue

            key_string = str(key)

            if current_memory_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD current_memory_proj.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "MUDD checkpoint model_channels "
                        "与当前模型不一致："
                        f"checkpoint={tensor.shape[1]}, "
                        f"model={model_channels}"
                    )

                memory_channels.add(
                    int(tensor.shape[0])
                )

                saw_core_base_weight = True

            elif shallow_memory_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD shallow_memory_proj.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "MUDD shallow_memory_proj 的输入 "
                        "channel 与当前模型不一致："
                        f"{tensor.shape[1]} != "
                        f"{model_channels}"
                    )

                memory_channels.add(
                    int(tensor.shape[0])
                )

                saw_core_base_weight = True

            elif memory_merge_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD memory_merge.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                merge_output = int(
                    tensor.shape[0]
                )

                merge_input = int(
                    tensor.shape[1]
                )

                if merge_input != 3 * merge_output:
                    raise RuntimeError(
                        "MUDD memory_merge.weight "
                        "形状不符合 3C -> C："
                        f"{tuple(tensor.shape)}"
                    )

                memory_channels.add(
                    merge_output
                )

                saw_core_base_weight = True

            elif output_proj_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD output_proj.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[0]) != model_channels:
                    raise RuntimeError(
                        "MUDD output_proj 输出 channel "
                        "与当前模型不一致："
                        f"{tensor.shape[0]} != "
                        f"{model_channels}"
                    )

                memory_channels.add(
                    int(tensor.shape[1])
                )

                saw_core_base_weight = True

            unit_match = memory_unit_pattern.search(
                key_string
            )

            if unit_match is not None:
                memory_unit_indices.add(
                    int(
                        unit_match.group(1)
                    )
                )

            depthwise_match = (
                memory_depthwise_pattern.search(
                    key_string
                )
            )

            if depthwise_match is not None:
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD memory depthwise.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                memory_channel = int(
                    tensor.shape[0]
                )

                if int(tensor.shape[1]) != 1:
                    raise RuntimeError(
                        "MUDD memory depthwise.weight "
                        "不是 depthwise Conv3d："
                        f"{tuple(tensor.shape)}"
                    )

                memory_channels.add(
                    memory_channel
                )

                temporal_kernel_sizes.add(
                    int(tensor.shape[2])
                )

                spatial_h = int(
                    tensor.shape[3]
                )

                spatial_w = int(
                    tensor.shape[4]
                )

                if spatial_h != spatial_w:
                    raise RuntimeError(
                        "MUDD memory spatial kernel "
                        "不是正方形："
                        f"{tuple(tensor.shape)}"
                    )

                spatial_kernel_sizes.add(
                    spatial_h
                )

                saw_any_memory_unit_weight = True

            if detail_in_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD detail_in_proj.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "MUDD detail_in_proj 输入 channel "
                        "与当前模型不一致："
                        f"{tensor.shape[1]} != "
                        f"{model_channels}"
                    )

                detail_channels.add(
                    int(tensor.shape[0])
                )

                saw_detail_base_weight = True

            if detail_depthwise_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD detail depthwise.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != 1:
                    raise RuntimeError(
                        "MUDD detail depthwise.weight "
                        "不是 depthwise Conv3d："
                        f"{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[2]) != 1:
                    raise RuntimeError(
                        "MUDD detail temporal kernel "
                        "必须为 1："
                        f"{tuple(tensor.shape)}"
                    )

                detail_channels.add(
                    int(tensor.shape[0])
                )

                spatial_h = int(
                    tensor.shape[3]
                )

                spatial_w = int(
                    tensor.shape[4]
                )

                if spatial_h != spatial_w:
                    raise RuntimeError(
                        "MUDD detail spatial kernel "
                        "不是正方形："
                        f"{tuple(tensor.shape)}"
                    )

                spatial_kernel_sizes.add(
                    spatial_h
                )

                saw_detail_base_weight = True

            if detail_to_memory_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "MUDD detail_to_memory.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                memory_channels.add(
                    int(tensor.shape[0])
                )

                detail_channels.add(
                    int(tensor.shape[1])
                )

                saw_detail_base_weight = True

        if len(memory_channels) > 1:
            raise RuntimeError(
                "同一个 MUDD checkpoint 包含多个 "
                "memory_channels："
                f"{sorted(memory_channels)}"
            )

        if memory_channels:
            memory_channel = next(
                iter(memory_channels)
            )

            config.memory_channels_override = (
                memory_channel
            )

            config.memory_ratio = (
                float(memory_channel)
                / float(model_channels)
            )

        if len(detail_channels) > 1:
            raise RuntimeError(
                "同一个 MUDD checkpoint 包含多个 "
                "detail_channels："
                f"{sorted(detail_channels)}"
            )

        if detail_channels:
            detail_channel = next(
                iter(detail_channels)
            )

            config.detail_channels_override = (
                detail_channel
            )

            config.detail_ratio = (
                float(detail_channel)
                / float(model_channels)
            )

            config.use_detail_branch = True
        elif saw_core_base_weight:
            # 当 source_state_dict 明确包含完整 core base weight，
            # 但完全没有 detail base weight 时，可推断训练时关闭了
            # detail branch。
            config.use_detail_branch = bool(
                saw_detail_base_weight
            )

        if memory_unit_indices:
            config.memory_depth = (
                max(memory_unit_indices) + 1
            )
        elif (
            saw_core_base_weight
            and not saw_any_memory_unit_weight
        ):
            config.memory_depth = 0

        if len(spatial_kernel_sizes) > 1:
            raise RuntimeError(
                "同一个 MUDD checkpoint 包含多个 "
                "spatial kernel size："
                f"{sorted(spatial_kernel_sizes)}"
            )

        if spatial_kernel_sizes:
            config.spatial_kernel_size = next(
                iter(spatial_kernel_sizes)
            )

        if len(temporal_kernel_sizes) > 1:
            raise RuntimeError(
                "同一个 MUDD checkpoint 包含多个 "
                "temporal kernel size："
                f"{sorted(temporal_kernel_sizes)}"
            )

        if temporal_kernel_sizes:
            config.temporal_kernel_size = next(
                iter(temporal_kernel_sizes)
            )

        return config

    @classmethod
    def install(
        cls,
        model: nn.Module,
        source_keys: Optional[
            Sequence[str]
        ] = None,
        source_state_dict: Optional[
            Dict[str, torch.Tensor]
        ] = None,
    ) -> nn.Module:
        """
        原地安装 MUDD-Former V2。

        source_keys:
            可来自基础模型或 LoRA，用于推断安装 block。

        source_state_dict:
            如果调用方能够提供完整参数字典，可进一步推断
            memory_channels、detail_channels、memory_depth 和 kernel。
        """

        existing_mutation_id = getattr(
            model,
            "_anima_mutation_id",
            None,
        )

        if existing_mutation_id is not None:
            if (
                existing_mutation_id
                == cls.MUTATION_ID
            ):
                return model

            raise RuntimeError(
                "当前模型已安装其他 Mutation："
                f"{existing_mutation_id}，不能再次安装 "
                f"{cls.MUTATION_ID}"
            )

        if not hasattr(model, "blocks"):
            raise AttributeError(
                "Anima diffusion_model 不存在 blocks 属性"
            )

        if not hasattr(
            model,
            "model_channels",
        ):
            raise AttributeError(
                "Anima diffusion_model 不存在 "
                "model_channels 属性"
            )

        num_blocks = len(
            model.blocks
        )

        if num_blocks <= 0:
            raise RuntimeError(
                "Anima diffusion_model.blocks 为空"
            )

        all_source_keys = list(
            source_keys or []
        )

        if source_state_dict:
            all_source_keys.extend(
                source_state_dict.keys()
            )

        mudd_block_indices = (
            cls._infer_indices_from_keys(
                all_source_keys
            )
        )

        if not mudd_block_indices:
            mudd_block_indices = (
                build_every_n_block_indices(
                    num_blocks=num_blocks,
                    stride=(
                        cls.DEFAULT_CONFIG
                        .block_stride
                    ),
                    include_last_partial=False,
                )
            )

        if not mudd_block_indices:
            raise RuntimeError(
                "没有可安装的 MUDD block。"
            )

        runtime_config = (
            cls._infer_config_from_state_dict(
                source_state_dict,
                model_channels=int(
                    model.model_channels
                ),
            )
        )

        reference_parameter = (
            _find_model_reference_parameter(
                model
            )
        )

        selected_indices = set(
            int(index)
            for index in mudd_block_indices
        )

        for block_index in sorted(
            selected_indices
        ):
            if not (
                0
                <= block_index
                < num_blocks
            ):
                raise ValueError(
                    "无效的 MUDD block index："
                    f"{block_index}，总 block 数="
                    f"{num_blocks}"
                )

            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.MODULE_NAMESPACE,
            ):
                graft = getattr(
                    block,
                    cls.MODULE_NAMESPACE,
                )

                if not isinstance(
                    graft,
                    MUDDFormerGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.MODULE_NAMESPACE}："
                        f"{type(graft).__name__}"
                    )
            else:
                graft = MUDDFormerGraft(
                    model_channels=int(
                        model.model_channels
                    ),
                    config=runtime_config,
                    block_index=block_index,
                )

                if reference_parameter is not None:
                    reference_device = (
                        reference_parameter.device
                    )

                    reference_dtype = (
                        reference_parameter.dtype
                    )

                    if (
                        reference_device.type
                        == "meta"
                    ):
                        # 避免将新建实体参数直接转成 meta 后无法通过
                        # 常规 .to() 恢复。这里只同步 dtype，后续交给
                        # ComfyUI model patcher 调度设备。
                        graft.to(
                            dtype=reference_dtype
                        )
                    else:
                        graft.to(
                            device=reference_device,
                            dtype=reference_dtype,
                        )

                block.add_module(
                    cls.MODULE_NAMESPACE,
                    graft,
                )

        # Block 0 即使不是 MUDD block，也必须捕获初始 shallow
        # feature。其原始参数路径不会发生变化。
        blocks_to_wrap = set(
            selected_indices
        )

        blocks_to_wrap.add(0)

        for block_index in sorted(
            blocks_to_wrap
        ):
            block = model.blocks[
                block_index
            ]

            _install_mudd_block_forward(
                block=block,
                model=model,
                block_index=block_index,
                apply_mudd=(
                    block_index
                    in selected_indices
                ),
            )

        _install_model_runtime_context_forward(
            model
        )

        object.__setattr__(
            model,
            "mudd_config",
            runtime_config,
        )

        object.__setattr__(
            model,
            "mudd_block_indices",
            sorted(selected_indices),
        )

        # 同时提供较通用的 graft 属性，方便外部工具查询。
        object.__setattr__(
            model,
            "graft_config",
            runtime_config,
        )

        object.__setattr__(
            model,
            "graft_block_indices",
            sorted(selected_indices),
        )

        object.__setattr__(
            model,
            "_anima_mutation_id",
            cls.MUTATION_ID,
        )

        object.__setattr__(
            model,
            "_anima_mutation_display_name",
            cls.DISPLAY_NAME,
        )

        print(
            "✅ [SpatialGraftV2/MUDD] 已安装到 blocks: "
            f"{sorted(selected_indices)}"
        )

        print(
            "ℹ️ [SpatialGraftV2/MUDD] 运行配置: "
            f"model_channels={model.model_channels}, "
            f"memory_channels="
            f"{runtime_config.memory_channels_override}, "
            f"memory_pool={runtime_config.memory_pool}, "
            f"memory_depth={runtime_config.memory_depth}, "
            f"use_detail_branch="
            f"{runtime_config.use_detail_branch}, "
            f"detail_channels="
            f"{runtime_config.detail_channels_override}, "
            f"spatial_kernel="
            f"{runtime_config.spatial_kernel_size}, "
            f"temporal_kernel="
            f"{runtime_config.temporal_kernel_size}, "
            f"use_rms_norm="
            f"{runtime_config.use_rms_norm}"
        )

        if reference_parameter is not None:
            print(
                "ℹ️ [SpatialGraftV2/MUDD] 新增模块参考精度: "
                f"dtype={reference_parameter.dtype}, "
                f"device={reference_parameter.device}"
            )

        return model


__all__ = [
    "GraftedAnima",
    "MUDDGraftRuntimeConfig",
    "MUDDFormerGraft",
    "MUDDMemoryUnit",
    "MUDDDetailUnit",
    "SimpleRMSNorm",
    "ChannelNorm3d",
    "build_every_n_block_indices",
]
