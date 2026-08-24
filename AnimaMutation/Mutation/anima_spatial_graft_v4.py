# coding=utf-8
# Mutation/anima_spatial_graft_v3.py
#
# Anima Spatial Graft V3
#
# Hybrid MUDD-Former + low-resolution AttnRes runtime mutation.
#
# 该文件只包含 ComfyUI 推理所需内容：
#
#   1. 复用 V2 MUDD-Former 推理模块；
#   2. 新增低分辨率 recurrent AttnRes 模块；
#   3. 原版 Anima block/model forward 的原地安装；
#   4. checkpoint / LoRA 架构检测；
#   5. 根据 checkpoint 权重形状恢复运行配置；
#   6. 严格的 dtype/device/shape/finite 检查。
#
# 原始 Anima 参数路径保持不变。
#
# 新增参数路径：
#
#     blocks.{index}.mudd_graft.*
#     blocks.{index}.spatial_graft.*
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
# 复用 Spatial Graft V2 / MUDD 实现
# ============================================================

import sys
from pathlib import Path

# 仅把当前Mutation文件夹加入搜索路径
curr = Path(__file__).parent
if str(curr) not in sys.path:
    sys.path.insert(0, str(curr))

# 同目录直接导入，不带任何 .
from anima_spatial_graft_v2 import (
    GraftedAnima as V2GraftedAnima,
    MUDDGraftRuntimeConfig,
    MUDDFormerGraft,
    MUDDMemoryUnit,
    MUDDDetailUnit,
    SimpleRMSNorm,
    ChannelNorm3d,
    build_every_n_block_indices,
)


# ============================================================
# AttnRes 配置
# ============================================================


@dataclass
class AttnResGraftRuntimeConfig:
    """
    低分辨率 recurrent AttnRes 推理配置。

    默认值与训练架构 anima_grafted3.py 保持一致。

    attention_channels_override 是推理恢复字段：

    - 训练配置通过 attention_ratio 计算 channel；
    - 推理时如果 checkpoint 中存在完整权重，可以从权重形状恢复
      精确 attention channel；
    - 该字段本身不会产生额外参数。
    """

    enabled: bool = True

    # 每组连续非 MUDD blocks 的大小。
    group_size: int = 3

    # 是否处理每个连续区段末尾不足 group_size 的部分。
    include_partial_group: bool = False

    # start / middle / end。
    placement: str = "middle"

    attention_ratio: float = 0.0625

    # 从完整 checkpoint 权重恢复的精确 channel。
    attention_channels_override: Optional[int] = None

    memory_pool: int = 4

    max_memory_tokens: int = 128

    max_temporal_tokens: int = 8

    num_heads: int = 4

    memory_depth: int = 1

    spatial_kernel_size: int = 3

    temporal_kernel_size: int = 1

    dropout: float = 0.0

    attention_layer_scale_init: float = 0.1

    memory_layer_scale_init: float = 0.1

    branch_scale_init: float = 1.0

    memory_update_bias: float = -1.5

    use_framewise_timestep: bool = True

    use_rms_norm: bool = True


# ============================================================
# dtype / device / validation 工具
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
    查找新增模块应该跟随的 dtype/device。

    优先选择 x_embedder，因为它通常最能代表 diffusion model
    实际使用的设备和计算精度。
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


def _cast_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        tensor.dtype != dtype
        or tensor.device != device
    ):
        tensor = tensor.to(
            device=device,
            dtype=dtype,
        )

    return tensor


def _require_same_device(
    tensor: torch.Tensor,
    expected_device: torch.device,
    tensor_name: str,
):
    if tensor.device != expected_device:
        raise RuntimeError(
            f"[SpatialGraftV3/AttnRes] {tensor_name} 与新增模块"
            "不在同一设备："
            f"tensor={tensor.device}, module={expected_device}。\n"
            "这通常表示 ComfyUI model patcher 或 block swap 没有"
            "将 spatial_graft 一起移动到当前 block 的设备。"
        )


def _check_finite(
    tensor: torch.Tensor,
    tensor_name: str,
):
    if not tensor.is_floating_point():
        return

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"[SpatialGraftV3/AttnRes] {tensor_name} 中检测到 "
            "NaN/Inf。已停止推理，以避免产生纯黑图或损坏输出。"
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


def _move_new_module_like_model(
    module: nn.Module,
    reference_parameter: Optional[nn.Parameter],
):
    if reference_parameter is None:
        return

    reference_dtype = reference_parameter.dtype
    reference_device = reference_parameter.device

    if reference_device.type == "meta":
        # 不把新建实体参数转成 meta，否则普通 .to() 无法恢复。
        module.to(
            dtype=reference_dtype
        )
    else:
        module.to(
            device=reference_device,
            dtype=reference_dtype,
        )


# ============================================================
# AttnRes 低分辨率 Memory Unit
# ============================================================


class AttnResMemoryUnit(nn.Module):
    """
    低分辨率 AttnRes memory convolutional FFN。

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

        if channels <= 0:
            raise ValueError(
                "AttnResMemoryUnit channels 必须大于 0"
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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.depthwise.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.pointwise_in.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.pointwise_out.weight,
            a=math.sqrt(5),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "AttnResMemoryUnit 输入必须是 [B,C,T,H,W]，"
                f"实际为 {tuple(x.shape)}"
            )

        compute_dtype = self.depthwise.weight.dtype
        compute_device = self.depthwise.weight.device

        _require_same_device(
            x,
            compute_device,
            "AttnResMemoryUnit 输入",
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

        layer_scale = self.layer_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
            1,
        )

        output = (
            residual
            + layer_scale * branch
        )

        return _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )


# ============================================================
# Compressed AttnRes Attention
# ============================================================


class CompressedAttnResAttention(nn.Module):
    """
    压缩 landmark 空间中的 QK-normalized multi-head attention。

    输入：

        query:   [B,Lq,C]
        context: [B,Lk,C]

    不会创建全分辨率图像 token attention matrix。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        channels = int(channels)
        num_heads = int(num_heads)

        if channels <= 0:
            raise ValueError(
                "AttnRes attention channels 必须大于 0"
            )

        if num_heads <= 0:
            raise ValueError(
                "AttnRes num_heads 必须大于 0"
            )

        if channels % num_heads != 0:
            raise ValueError(
                f"AttnRes channels={channels} 不能被 "
                f"num_heads={num_heads} 整除"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )

        self.k_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )

        self.v_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )

        self.o_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )

        self.q_norm = SimpleRMSNorm(
            self.head_dim,
            eps=1e-6,
            elementwise_affine=False,
        )

        self.k_norm = SimpleRMSNorm(
            self.head_dim,
            eps=1e-6,
            elementwise_affine=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(
            self.channels
        )

        nn.init.trunc_normal_(
            self.q_proj.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.k_proj.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.v_proj.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.o_proj.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError(
                "AttnRes query 必须是 [B,L,C]，"
                f"实际为 {tuple(query.shape)}"
            )

        if context.ndim != 3:
            raise ValueError(
                "AttnRes context 必须是 [B,L,C]，"
                f"实际为 {tuple(context.shape)}"
            )

        batch_size, query_length, channels = (
            query.shape
        )

        (
            context_batch,
            context_length,
            context_channels,
        ) = context.shape

        if batch_size != context_batch:
            raise ValueError(
                "AttnRes query/context batch 不一致："
                f"{batch_size} != {context_batch}"
            )

        if (
            channels != self.channels
            or context_channels != self.channels
        ):
            raise ValueError(
                "AttnRes query/context channel 不一致："
                f"query={channels}, context={context_channels}, "
                f"expected={self.channels}"
            )

        compute_dtype = self.q_proj.weight.dtype
        compute_device = self.q_proj.weight.device

        _require_same_device(
            query,
            compute_device,
            "AttnRes query",
        )

        _require_same_device(
            context,
            compute_device,
            "AttnRes context",
        )

        query = _cast_tensor(
            query,
            dtype=compute_dtype,
            device=compute_device,
        )

        context = _cast_tensor(
            context,
            dtype=compute_dtype,
            device=compute_device,
        )

        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = _cast_tensor(
            q,
            dtype=compute_dtype,
            device=compute_device,
        )

        k = _cast_tensor(
            k,
            dtype=compute_dtype,
            device=compute_device,
        )

        v = _cast_tensor(
            v,
            dtype=compute_dtype,
            device=compute_device,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = (
            self.dropout
            if self.training
            else 0.0
        )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

        output = output.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            query_length,
            self.channels,
        )

        output = _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )

        output = self.o_proj(output)

        return _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )


# ============================================================
# AttnRes Graft
# ============================================================


class AttnResGraft(nn.Module):
    """
    低分辨率 recurrent residual-attention graft。

    输入：

        x_B_T_H_W_D:
            当前 Anima hidden feature。

        shallow_B_T_H_W_D:
            block 0 之前的初始 patch embedding。

        previous_memory:
            前一个 spatial_graft 的 recurrent memory，
            形状为 [B,Ca,Tm,Hm,Wm]，或 None。

        timestep_embedding_B_T_D:
            Anima 已归一化的 timestep embedding。

    输出：

        output:
            当前 hidden feature 加 AttnRes residual。

        updated_memory:
            传给下一个 spatial_graft 的低分辨率 memory。
    """

    def __init__(
        self,
        model_channels: int,
        config: AttnResGraftRuntimeConfig,
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
            config.attention_channels_override
            is not None
        ):
            attention_channels = int(
                config.attention_channels_override
            )

            if attention_channels <= 0:
                raise ValueError(
                    "attention_channels_override 必须大于 0"
                )
        else:
            attention_channels = max(
                32,
                int(
                    round(
                        self.model_channels
                        * float(
                            config.attention_ratio
                        )
                    )
                ),
            )

            attention_channels = int(
                math.ceil(
                    attention_channels / 32
                ) * 32
            )

            attention_channels = min(
                attention_channels,
                self.model_channels,
            )

        requested_heads = min(
            int(config.num_heads),
            attention_channels,
        )

        if requested_heads <= 0:
            requested_heads = 1

        while (
            requested_heads > 1
            and attention_channels % requested_heads != 0
        ):
            requested_heads -= 1

        self.attention_channels = int(
            attention_channels
        )

        self.num_heads = int(
            requested_heads
        )

        if int(config.memory_pool) <= 0:
            raise ValueError(
                "AttnRes memory_pool 必须大于 0"
            )

        if int(config.max_memory_tokens) <= 0:
            raise ValueError(
                "AttnRes max_memory_tokens 必须大于 0"
            )

        if int(config.max_temporal_tokens) <= 0:
            raise ValueError(
                "AttnRes max_temporal_tokens 必须大于 0"
            )

        if int(config.memory_depth) < 0:
            raise ValueError(
                "AttnRes memory_depth 不能小于 0"
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

        self.current_proj = nn.Conv3d(
            self.model_channels,
            self.attention_channels,
            kernel_size=1,
            bias=False,
        )

        self.shallow_proj = nn.Conv3d(
            self.model_channels,
            self.attention_channels,
            kernel_size=1,
            bias=False,
        )

        self.position_mixer = nn.Conv3d(
            self.attention_channels,
            self.attention_channels,
            kernel_size=(
                int(config.temporal_kernel_size),
                int(config.spatial_kernel_size),
                int(config.spatial_kernel_size),
            ),
            stride=1,
            padding=(
                int(config.temporal_kernel_size) // 2,
                int(config.spatial_kernel_size) // 2,
                int(config.spatial_kernel_size) // 2,
            ),
            groups=self.attention_channels,
            bias=False,
        )

        self.attention = CompressedAttnResAttention(
            channels=self.attention_channels,
            num_heads=self.num_heads,
            dropout=config.dropout,
        )

        self.attention_layer_scale = nn.Parameter(
            torch.full(
                (self.attention_channels,),
                float(
                    config.attention_layer_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.memory_units = nn.ModuleList(
            [
                AttnResMemoryUnit(
                    channels=self.attention_channels,
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

        # 输出：
        #
        #   1. recurrent memory update logits；
        #   2. visible output strength；
        #   3. shallow context strength；
        #   4. previous-memory context strength。
        self.time_modulation = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.model_channels,
                self.attention_channels,
                bias=False,
            ),

            nn.SiLU(),

            nn.Linear(
                self.attention_channels,
                4 * self.attention_channels,
                bias=True,
            ),
        )

        self.output_proj = nn.Conv3d(
            self.attention_channels,
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
            self.current_proj.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.shallow_proj.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.position_mixer.weight,
            a=math.sqrt(5),
        )

        self.attention.reset_parameters()

        for unit in self.memory_units:
            unit.reset_parameters()

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

        # 初始 timestep modulation 保持常量。
        nn.init.zeros_(
            self.time_modulation[3].weight
        )

        nn.init.zeros_(
            self.time_modulation[3].bias
        )

        channels = self.attention_channels

        with torch.no_grad():
            self.time_modulation[3].bias[
                :channels
            ].fill_(
                float(
                    self.config.memory_update_bias
                )
            )

            self.time_modulation[3].bias[
                channels:
            ].zero_()

        # 未加载 V3 权重时，不改变原始 Anima 可见输出。
        nn.init.zeros_(
            self.output_proj.weight
        )

    def _initial_spatial_pool(
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

    def _calculate_budget_shape(
        self,
        temporal: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int]:
        max_tokens = max(
            1,
            int(
                self.config.max_memory_tokens
            ),
        )

        target_t = min(
            int(temporal),
            max(
                1,
                int(
                    self.config.max_temporal_tokens
                ),
            ),
        )

        spatial_budget = max(
            1,
            max_tokens // target_t,
        )

        if height * width <= spatial_budget:
            target_h = int(height)
            target_w = int(width)
        else:
            aspect_ratio = (
                float(height)
                / max(float(width), 1.0)
            )

            target_h = max(
                1,
                int(
                    math.sqrt(
                        spatial_budget
                        * aspect_ratio
                    )
                ),
            )

            target_w = max(
                1,
                spatial_budget // target_h,
            )

            target_h = min(
                int(height),
                target_h,
            )

            target_w = min(
                int(width),
                target_w,
            )

            while (
                target_t
                * target_h
                * target_w
                > max_tokens
            ):
                if (
                    target_h >= target_w
                    and target_h > 1
                ):
                    target_h -= 1
                elif target_w > 1:
                    target_w -= 1
                elif target_t > 1:
                    target_t -= 1
                else:
                    break

        return (
            max(1, int(target_t)),
            max(1, int(target_h)),
            max(1, int(target_w)),
        )

    def _pool_to_budget(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        input_device = x.device

        x = self._initial_spatial_pool(
            x
        )

        _, _, temporal, height, width = (
            x.shape
        )

        target_shape = (
            self._calculate_budget_shape(
                temporal,
                height,
                width,
            )
        )

        if target_shape != (
            temporal,
            height,
            width,
        ):
            x = F.adaptive_avg_pool3d(
                x,
                output_size=target_shape,
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
                "AttnRes timestep embedding 必须是 Tensor"
            )

        if timestep_embedding.ndim != 3:
            raise ValueError(
                "AttnRes timestep embedding 期望 [B,T,D]，"
                f"实际为 {tuple(timestep_embedding.shape)}"
            )

        if (
            timestep_embedding.shape[0]
            != batch_size
        ):
            raise ValueError(
                "AttnRes timestep batch 不一致："
                f"{timestep_embedding.shape[0]} != "
                f"{batch_size}"
            )

        if (
            timestep_embedding.shape[-1]
            != self.model_channels
        ):
            raise ValueError(
                "AttnRes timestep channel 不一致："
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
                    "AttnRes timestep 帧数不一致："
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

    @staticmethod
    def _resize_temporal_gate(
        gate: torch.Tensor,
        target_t: int,
    ) -> torch.Tensor:
        # gate: [B,C,T,1,1]
        if gate.shape[2] == target_t:
            return gate

        input_dtype = gate.dtype
        input_device = gate.device

        gate = F.adaptive_avg_pool3d(
            gate,
            output_size=(
                int(target_t),
                1,
                1,
            ),
        )

        return _cast_tensor(
            gate,
            dtype=input_dtype,
            device=input_device,
        )

    def _prepare_previous_memory(
        self,
        previous_memory: Optional[torch.Tensor],
        shallow_memory: torch.Tensor,
        current_memory: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = current_memory.dtype
        compute_device = current_memory.device

        if previous_memory is None:
            return shallow_memory

        if not torch.is_tensor(
            previous_memory
        ):
            raise TypeError(
                "AttnRes previous_memory 必须是 Tensor 或 None"
            )

        if previous_memory.ndim != 5:
            raise ValueError(
                "AttnRes previous_memory 期望 [B,C,T,H,W]，"
                f"实际为 {tuple(previous_memory.shape)}"
            )

        if (
            previous_memory.shape[0]
            != current_memory.shape[0]
        ):
            raise ValueError(
                "AttnRes previous_memory batch 不一致："
                f"{previous_memory.shape[0]} != "
                f"{current_memory.shape[0]}"
            )

        if (
            previous_memory.shape[1]
            != self.attention_channels
        ):
            raise ValueError(
                "AttnRes previous_memory channel 不一致："
                f"{previous_memory.shape[1]} != "
                f"{self.attention_channels}"
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

        return previous_memory

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
                "AttnRes 当前特征必须是 Tensor"
            )

        if not torch.is_tensor(
            shallow_B_T_H_W_D
        ):
            raise TypeError(
                "AttnRes shallow feature 必须是 Tensor"
            )

        if x_B_T_H_W_D.ndim != 5:
            raise ValueError(
                "AttnRes 当前特征期望 [B,T,H,W,D]，"
                f"实际为 {tuple(x_B_T_H_W_D.shape)}"
            )

        if shallow_B_T_H_W_D.ndim != 5:
            raise ValueError(
                "AttnRes shallow feature 期望 [B,T,H,W,D]，"
                f"实际为 {tuple(shallow_B_T_H_W_D.shape)}"
            )

        if (
            shallow_B_T_H_W_D.shape
            != x_B_T_H_W_D.shape
        ):
            raise ValueError(
                "AttnRes shallow/current shape 不一致："
                f"{tuple(shallow_B_T_H_W_D.shape)} != "
                f"{tuple(x_B_T_H_W_D.shape)}"
            )

        compute_dtype = self.current_proj.weight.dtype
        compute_device = self.current_proj.weight.device

        _require_same_device(
            x_B_T_H_W_D,
            compute_device,
            "AttnRes 当前特征",
        )

        _require_same_device(
            shallow_B_T_H_W_D,
            compute_device,
            "AttnRes shallow feature",
        )

        _require_same_device(
            timestep_embedding_B_T_D,
            compute_device,
            "AttnRes timestep embedding",
        )

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
                "AttnRes 输入 channel 不匹配："
                f"{channels} != {self.model_channels}"
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

        current_memory = self.current_proj(
            current_3d
        )

        shallow_memory = self.shallow_proj(
            shallow_3d
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

        current_memory = self._pool_to_budget(
            current_memory
        )

        shallow_memory = self._pool_to_budget(
            shallow_memory
        )

        if (
            shallow_memory.shape[-3:]
            != current_memory.shape[-3:]
        ):
            shallow_memory = F.interpolate(
                shallow_memory,
                size=current_memory.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

            shallow_memory = _cast_tensor(
                shallow_memory,
                dtype=compute_dtype,
                device=compute_device,
            )

        previous_memory = (
            self._prepare_previous_memory(
                previous_memory,
                shallow_memory,
                current_memory,
            )
        )

        current_positioned = (
            current_memory
            + self.position_mixer(
                current_memory
            )
        )

        shallow_positioned = (
            shallow_memory
            + self.position_mixer(
                shallow_memory
            )
        )

        previous_positioned = (
            previous_memory
            + self.position_mixer(
                previous_memory
            )
        )

        current_positioned = _cast_tensor(
            current_positioned,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_positioned = _cast_tensor(
            shallow_positioned,
            dtype=compute_dtype,
            device=compute_device,
        )

        previous_positioned = _cast_tensor(
            previous_positioned,
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
            previous_gate,
        ) = modulation.chunk(
            4,
            dim=-1,
        )

        # [B,T,C] -> [B,C,T,1,1]
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

        previous_gate = previous_gate.permute(
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

        target_t = int(
            current_memory.shape[2]
        )

        update_logits = (
            self._resize_temporal_gate(
                update_logits,
                target_t,
            )
        )

        output_gate = (
            self._resize_temporal_gate(
                output_gate,
                target_t,
            )
        )

        shallow_gate = (
            self._resize_temporal_gate(
                shallow_gate,
                target_t,
            )
        )

        previous_gate = (
            self._resize_temporal_gate(
                previous_gate,
                target_t,
            )
        )

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

        previous_gate = _cast_tensor(
            previous_gate,
            dtype=compute_dtype,
            device=compute_device,
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

        update_strength = torch.sigmoid(
            update_logits
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

        previous_strength = (
            one
            + half * torch.tanh(
                previous_gate
            )
        )

        update_strength = _cast_tensor(
            update_strength,
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

        previous_strength = _cast_tensor(
            previous_strength,
            dtype=compute_dtype,
            device=compute_device,
        )

        current_tokens = current_positioned.permute(
            0,
            2,
            3,
            4,
            1,
        ).reshape(
            batch_size,
            -1,
            self.attention_channels,
        )

        shallow_tokens = (
            shallow_positioned
            * shallow_strength
        ).permute(
            0,
            2,
            3,
            4,
            1,
        ).reshape(
            batch_size,
            -1,
            self.attention_channels,
        )

        previous_tokens = (
            previous_positioned
            * previous_strength
        ).permute(
            0,
            2,
            3,
            4,
            1,
        ).reshape(
            batch_size,
            -1,
            self.attention_channels,
        )

        context_tokens = torch.cat(
            [
                current_tokens,
                shallow_tokens,
                previous_tokens,
            ],
            dim=1,
        )

        context_tokens = _cast_tensor(
            context_tokens,
            dtype=compute_dtype,
            device=compute_device,
        )

        attention_output = self.attention(
            current_tokens,
            context_tokens,
        )

        attention_output = attention_output.reshape(
            batch_size,
            current_memory.shape[2],
            current_memory.shape[3],
            current_memory.shape[4],
            self.attention_channels,
        ).permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        attention_output = _cast_tensor(
            attention_output,
            dtype=compute_dtype,
            device=compute_device,
        )

        attention_scale = (
            self.attention_layer_scale.to(
                device=compute_device,
                dtype=compute_dtype,
            ).reshape(
                1,
                -1,
                1,
                1,
                1,
            )
        )

        memory_candidate = (
            current_memory
            + attention_scale
            * attention_output
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
            + update_strength
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

        visible_memory = (
            updated_memory
            * output_strength
        )

        visible_memory = _cast_tensor(
            visible_memory,
            dtype=compute_dtype,
            device=compute_device,
        )

        restored_memory = F.interpolate(
            visible_memory,
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

        output_feature = self.output_proj(
            restored_memory
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

        output = _cast_tensor(
            output,
            dtype=compute_dtype,
            device=compute_device,
        )

        _check_finite(
            output,
            "AttnRes 输出",
        )

        _check_finite(
            updated_memory,
            "AttnRes updated_memory",
        )

        return output, updated_memory


# ============================================================
# AttnRes block placement
# ============================================================


def _select_group_position(
    group: Sequence[int],
    placement: str,
) -> int:
    if not group:
        raise ValueError(
            "不能从空 block group 中选择 AttnRes block"
        )

    placement = str(
        placement
    ).lower()

    if placement == "start":
        return int(group[0])

    if placement == "end":
        return int(group[-1])

    if placement == "middle":
        # 长度 3 -> index 1。
        # 长度 2 -> index 1，与训练端常见 int(len/2) 一致。
        return int(
            group[len(group) // 2]
        )

    raise ValueError(
        "AttnRes placement 必须为 start/middle/end，"
        f"实际为 {placement}"
    )


def build_attnres_block_indices(
    num_blocks: int,
    mudd_block_indices: Sequence[int],
    group_size: int = 3,
    include_partial_group: bool = False,
    placement: str = "middle",
) -> List[int]:
    """
    在连续非 MUDD blocks 中建立 placement group。

    示例：

        num_blocks = 28
        mudd = [3,7,11,15,19,23,27]
        group_size = 3
        placement = middle

    返回：

        [1,5,9,13,17,21,25]
    """

    num_blocks = int(
        num_blocks
    )

    group_size = int(
        group_size
    )

    if num_blocks <= 0:
        return []

    if group_size <= 0:
        raise ValueError(
            "AttnRes group_size 必须大于 0"
        )

    mudd_set = {
        int(index)
        for index in mudd_block_indices
    }

    selected: List[int] = []
    current_segment: List[int] = []

    def flush_segment():
        nonlocal current_segment

        if not current_segment:
            return

        start = 0

        while (
            start + group_size
            <= len(current_segment)
        ):
            group = current_segment[
                start:
                start + group_size
            ]

            selected.append(
                _select_group_position(
                    group,
                    placement,
                )
            )

            start += group_size

        remainder = current_segment[
            start:
        ]

        if (
            include_partial_group
            and remainder
        ):
            selected.append(
                _select_group_position(
                    remainder,
                    placement,
                )
            )

        current_segment = []

    for block_index in range(
        num_blocks
    ):
        if block_index in mudd_set:
            flush_segment()
            continue

        current_segment.append(
            block_index
        )

    flush_segment()

    return sorted(
        set(selected)
    )


# ============================================================
# Runtime context / block forward 安装
# ============================================================


def _get_or_create_runtime_context(
    model: nn.Module,
) -> Dict[str, Optional[torch.Tensor]]:
    context = getattr(
        model,
        "_hybrid_graft_runtime_context",
        None,
    )

    if context is None:
        context = {
            "shallow": None,
            "mudd_memory": None,
            "attnres_memory": None,
        }

        object.__setattr__(
            model,
            "_hybrid_graft_runtime_context",
            context,
        )

    return context


def _extract_primary_tensor(
    output,
) -> Tuple[torch.Tensor, Optional[str]]:
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
        "[SpatialGraftV3] 原始 Anima block.forward 返回了"
        f"不支持的类型：{type(output).__name__}。\n"
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
        f"未知 block 输出容器类型：{container_type}"
    )


def _install_hybrid_block_forward(
    block: nn.Module,
    model: nn.Module,
    block_index: int,
    apply_mudd: bool,
    apply_attnres: bool,
):
    """
    原地修改 block.forward。

    执行顺序：

        original Anima block
            -> MUDD（如果当前 block 被选中）
            -> AttnRes（如果当前 block 被选中）

    默认配置下 MUDD 与 AttnRes block 不重叠，但这里允许 checkpoint
    明确指定重叠 block。
    """

    if getattr(
        block,
        "_hybrid_graft_forward_installed",
        False,
    ):
        if apply_mudd:
            object.__setattr__(
                block,
                "_hybrid_apply_mudd",
                True,
            )

        if apply_attnres:
            object.__setattr__(
                block,
                "_hybrid_apply_attnres",
                True,
            )

        return

    if apply_mudd and not hasattr(
        block,
        "mudd_graft",
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 mudd_graft"
        )

    if apply_attnres and not hasattr(
        block,
        "spatial_graft",
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 spatial_graft"
        )

    # V3 必须安装在未被其他 mutation 包装的原版 block 上。
    if hasattr(
        block,
        "_original_anima_forward",
    ):
        raise RuntimeError(
            f"Block {block_index} 已被其他 Anima Mutation 修改，"
            "不能在同一模型实例上继续安装 Spatial Graft V3。"
        )

    original_forward = block.forward

    object.__setattr__(
        block,
        "_original_anima_forward",
        original_forward,
    )

    object.__setattr__(
        block,
        "_hybrid_owner_model_ref",
        weakref.ref(model),
    )

    object.__setattr__(
        block,
        "_hybrid_runtime_block_index",
        int(block_index),
    )

    object.__setattr__(
        block,
        "_hybrid_apply_mudd",
        bool(apply_mudd),
    )

    object.__setattr__(
        block,
        "_hybrid_apply_attnres",
        bool(apply_attnres),
    )

    def hybrid_grafted_forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        *args,
        **kwargs,
    ):
        owner_reference = getattr(
            self,
            "_hybrid_owner_model_ref",
            None,
        )

        owner_model = (
            owner_reference()
            if owner_reference is not None
            else None
        )

        if owner_model is None:
            raise RuntimeError(
                "[SpatialGraftV3] 无法取得所属 Anima model，"
                "runtime context 已失效。"
            )

        context = _get_or_create_runtime_context(
            owner_model
        )

        runtime_block_index = int(
            getattr(
                self,
                "_hybrid_runtime_block_index",
                -1,
            )
        )

        # 每次完整 forward 进入 block 0 时重置所有 recurrent memory。
        if runtime_block_index == 0:
            if not torch.is_tensor(
                x_B_T_H_W_D
            ):
                raise TypeError(
                    "[SpatialGraftV3] Block 0 输入不是 Tensor"
                )

            context["shallow"] = (
                x_B_T_H_W_D
            )

            context["mudd_memory"] = None
            context["attnres_memory"] = None

        original_output = (
            self._original_anima_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                *args,
                **kwargs,
            )
        )

        apply_mudd_now = bool(
            getattr(
                self,
                "_hybrid_apply_mudd",
                False,
            )
        )

        apply_attnres_now = bool(
            getattr(
                self,
                "_hybrid_apply_attnres",
                False,
            )
        )

        if (
            not apply_mudd_now
            and not apply_attnres_now
        ):
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
                "[SpatialGraftV3] 未捕获 shallow feature。\n"
                "Anima blocks 可能没有从 block 0 顺序执行，或者"
                "当前运行时绕过了标准 Anima block forward。"
            )

        if apply_mudd_now:
            hidden_states, mudd_memory = (
                self.mudd_graft(
                    hidden_states,
                    shallow_feature,
                    context.get(
                        "mudd_memory"
                    ),
                    emb_B_T_D,
                )
            )

            context["mudd_memory"] = (
                mudd_memory
            )

        if apply_attnres_now:
            hidden_states, attnres_memory = (
                self.spatial_graft(
                    hidden_states,
                    shallow_feature,
                    context.get(
                        "attnres_memory"
                    ),
                    emb_B_T_D,
                )
            )

            context["attnres_memory"] = (
                attnres_memory
            )

        return _replace_primary_tensor(
            original_output,
            hidden_states,
            container_type,
        )

    block.forward = MethodType(
        hybrid_grafted_forward,
        block,
    )

    object.__setattr__(
        block,
        "_hybrid_graft_forward_installed",
        True,
    )


def _install_model_runtime_context_forward(
    model: nn.Module,
):
    """
    包装顶层 forward，只负责创建和清理 runtime context。

    不复制 Anima 顶层逻辑，因此能够继续兼容：

    - attention backend；
    - block swap；
    - LLM adapter；
    - padding mask；
    - ComfyUI 对 Anima forward 的其他兼容修改。
    """

    if getattr(
        model,
        "_hybrid_model_forward_installed",
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
    elif hasattr(
        model,
        "forward",
    ):
        forward_name = "forward"
    else:
        raise AttributeError(
            "Anima model 不存在 forward_mini_train_dit "
            "或 forward"
        )

    original_forward = getattr(
        model,
        forward_name,
    )

    object.__setattr__(
        model,
        "_original_anima_hybrid_model_forward",
        original_forward,
    )

    object.__setattr__(
        model,
        "_hybrid_wrapped_forward_name",
        forward_name,
    )

    sentinel = object()

    def hybrid_model_forward(
        self,
        *args,
        **kwargs,
    ):
        previous_context = getattr(
            self,
            "_hybrid_graft_runtime_context",
            sentinel,
        )

        object.__setattr__(
            self,
            "_hybrid_graft_runtime_context",
            {
                "shallow": None,
                "mudd_memory": None,
                "attnres_memory": None,
            },
        )

        try:
            return (
                self._original_anima_hybrid_model_forward(
                    *args,
                    **kwargs,
                )
            )
        finally:
            if previous_context is sentinel:
                try:
                    object.__delattr__(
                        self,
                        "_hybrid_graft_runtime_context",
                    )
                except AttributeError:
                    pass
            else:
                object.__setattr__(
                    self,
                    "_hybrid_graft_runtime_context",
                    previous_context,
                )

    setattr(
        model,
        forward_name,
        MethodType(
            hybrid_model_forward,
            model,
        ),
    )

    object.__setattr__(
        model,
        "_hybrid_model_forward_installed",
        True,
    )


# ============================================================
# Mutation 入口
# ============================================================


class GraftedAnima:
    """
    AnimaBaker Mutation 系统入口。

    该类不是 nn.Module，而是 V3 架构检测器和原地安装器。
    """

    MUTATION_API_VERSION = 1

    MUTATION_ID = "anima_spatial_graft_v3"

    DISPLAY_NAME = (
        "Anima Hybrid MUDD + AttnRes Spatial Graft V3"
    )

    # 两条真实模块参数路径：
    #
    # blocks.{index}.mudd_graft.*
    # blocks.{index}.spatial_graft.*
    MUDD_NAMESPACE = "mudd_graft"
    ATTNRES_NAMESPACE = "spatial_graft"

    # Mutation API v1 只接受一个非空字符串。
    # spatial_graft 是 V3 的主识别命名空间。
    MODULE_NAMESPACE = ATTNRES_NAMESPACE

    DEFAULT_MUDD_CONFIG = (
        MUDDGraftRuntimeConfig(
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
    )

    DEFAULT_ATTNRES_CONFIG = (
        AttnResGraftRuntimeConfig(
            enabled=True,
            group_size=3,
            include_partial_group=False,
            placement="middle",
            attention_ratio=0.0625,
            attention_channels_override=None,
            memory_pool=4,
            max_memory_tokens=128,
            max_temporal_tokens=8,
            num_heads=4,
            memory_depth=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
            dropout=0.0,
            attention_layer_scale_init=0.1,
            memory_layer_scale_init=0.1,
            branch_scale_init=1.0,
            memory_update_bias=-1.5,
            use_framewise_timestep=True,
            use_rms_norm=True,
        )
    )

    @classmethod
    def detect(
        cls,
        state_dict_keys,
    ) -> int:
        """
        检测 V3 hybrid 架构。

        评分设计：

        - 只有 mudd_graft：
          不判定为 V3，交给 V2；
        - 只有 spatial_graft：
          可判定为 AttnRes V3，但分数略低于完整 hybrid；
        - 同时存在 mudd_graft + spatial_graft：
          明确判定为 V3，分数高于 V2 的 105；
        - 含 MUTATION_ID：
          最高优先级。
        """

        keys = [
            str(key).lower()
            for key in (
                state_dict_keys or []
            )
        ]

        if any(
            cls.MUTATION_ID in key
            for key in keys
        ):
            return 130

        has_mudd = any(
            (
                ".mudd_graft." in key
                or "_mudd_graft_" in key
                or key.startswith(
                    "mudd_graft."
                )
            )
            for key in keys
        )

        has_spatial = any(
            (
                ".spatial_graft." in key
                or "_spatial_graft_" in key
                or key.startswith(
                    "spatial_graft."
                )
            )
            for key in keys
        )

        has_attnres_attention = any(
            (
                ".spatial_graft.attention.q_proj." in key
                or "_spatial_graft_attention_q_proj_" in key
                or ".spatial_graft.attention_layer_scale" in key
                or "_spatial_graft_attention_layer_scale" in key
            )
            for key in keys
        )

        has_position_mixer = any(
            (
                ".spatial_graft.position_mixer." in key
                or "_spatial_graft_position_mixer_" in key
            )
            for key in keys
        )

        has_previous_gate_modulation = any(
            (
                ".spatial_graft.time_modulation.3." in key
                or "_spatial_graft_time_modulation_3_" in key
            )
            for key in keys
        )

        # 完整 Hybrid 是 V3 的最强结构标记。
        if (
            has_mudd
            and has_spatial
            and (
                has_attnres_attention
                or has_position_mixer
            )
        ):
            return 122

        if has_mudd and has_spatial:
            return 118

        # 只保存 AttnRes LoRA 时仍需安装完整 V3。
        if (
            has_spatial
            and has_attnres_attention
            and has_position_mixer
        ):
            return 114

        if (
            has_spatial
            and (
                has_attnres_attention
                or has_position_mixer
                or has_previous_gate_modulation
            )
        ):
            return 110

        if has_spatial:
            # 通用 spatial_graft 命名空间可能与其他旧变体冲突，
            # 因而不给 100+ 的明确 V3 分数。
            return 88

        # 只有 MUDD 时必须让 V2 接管。
        return 0

    @classmethod
    def is_mutation_key(
        cls,
        key,
    ) -> bool:
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

            or ".spatial_graft." in key_lower
            or key_lower.startswith(
                "spatial_graft."
            )
            or "_spatial_graft_" in key_lower
            or "spatial_graft." in key_lower
            or "spatial_graft_" in key_lower

            or cls.MUTATION_ID in key_lower
        )

    @classmethod
    def _infer_namespace_indices(
        cls,
        source_keys,
        namespace: str,
    ) -> List[int]:
        indices = set()

        escaped_namespace = re.escape(
            namespace
        )

        patterns = (
            re.compile(
                rf"(?:^|\.)blocks\.(\d+)"
                rf"\.{escaped_namespace}(?:\.|$)",
                re.IGNORECASE,
            ),

            re.compile(
                rf"(?:^|_)blocks_(\d+)"
                rf"_{escaped_namespace}(?:_|$)",
                re.IGNORECASE,
            ),
        )

        for key in source_keys or []:
            key_string = str(
                key
            )

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

        return sorted(
            indices
        )

    @classmethod
    def _infer_mudd_config(
        cls,
        source_state_dict: Optional[
            Dict[str, torch.Tensor]
        ],
        model_channels: int,
    ) -> MUDDGraftRuntimeConfig:
        """
        使用 V2 已验证的 checkpoint shape 推断逻辑。
        """

        if not source_state_dict:
            return copy.deepcopy(
                cls.DEFAULT_MUDD_CONFIG
            )

        return (
            V2GraftedAnima
            ._infer_config_from_state_dict(
                source_state_dict,
                model_channels=int(
                    model_channels
                ),
            )
        )

    @classmethod
    def _infer_attnres_config(
        cls,
        source_state_dict: Optional[
            Dict[str, torch.Tensor]
        ],
        model_channels: int,
    ) -> AttnResGraftRuntimeConfig:
        config = copy.deepcopy(
            cls.DEFAULT_ATTNRES_CONFIG
        )

        if not source_state_dict:
            return config

        model_channels = int(
            model_channels
        )

        attention_channels = set()
        memory_unit_indices = set()
        spatial_kernel_sizes = set()
        temporal_kernel_sizes = set()

        saw_core_weight = False
        saw_any_memory_unit_weight = False

        current_proj_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft"
            r"\.current_proj\.weight$",
            re.IGNORECASE,
        )

        shallow_proj_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft"
            r"\.shallow_proj\.weight$",
            re.IGNORECASE,
        )

        output_proj_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft"
            r"\.output_proj\.weight$",
            re.IGNORECASE,
        )

        position_mixer_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft"
            r"\.position_mixer\.weight$",
            re.IGNORECASE,
        )

        attention_projection_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft"
            r"\.attention\."
            r"(?:q_proj|k_proj|v_proj|o_proj)"
            r"\.weight$",
            re.IGNORECASE,
        )

        memory_unit_pattern = re.compile(
            r"\.spatial_graft\.memory_units\."
            r"(\d+)\.",
            re.IGNORECASE,
        )

        memory_depthwise_pattern = re.compile(
            r"\.spatial_graft\.memory_units\."
            r"(\d+)\.depthwise\.weight$",
            re.IGNORECASE,
        )

        time_first_pattern = re.compile(
            r"\.spatial_graft"
            r"\.time_modulation\.1\.weight$",
            re.IGNORECASE,
        )

        time_last_pattern = re.compile(
            r"\.spatial_graft"
            r"\.time_modulation\.3\.weight$",
            re.IGNORECASE,
        )

        for key, tensor in (
            source_state_dict.items()
        ):
            if not torch.is_tensor(
                tensor
            ):
                continue

            key_string = str(
                key
            )

            if current_proj_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "AttnRes current_proj.weight 维度异常："
                        f"{key_string} -> {tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "AttnRes current_proj 输入 channel 与"
                        "当前模型不一致："
                        f"{tensor.shape[1]} != {model_channels}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
                )

                saw_core_weight = True

            elif shallow_proj_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "AttnRes shallow_proj.weight 维度异常："
                        f"{key_string} -> {tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "AttnRes shallow_proj 输入 channel 与"
                        "当前模型不一致："
                        f"{tensor.shape[1]} != {model_channels}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
                )

                saw_core_weight = True

            elif output_proj_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "AttnRes output_proj.weight 维度异常："
                        f"{key_string} -> {tuple(tensor.shape)}"
                    )

                if int(tensor.shape[0]) != model_channels:
                    raise RuntimeError(
                        "AttnRes output_proj 输出 channel 与"
                        "当前模型不一致："
                        f"{tensor.shape[0]} != {model_channels}"
                    )

                attention_channels.add(
                    int(tensor.shape[1])
                )

                saw_core_weight = True

            elif position_mixer_pattern.search(
                key_string
            ):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "AttnRes position_mixer.weight 维度异常："
                        f"{key_string} -> {tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != 1:
                    raise RuntimeError(
                        "AttnRes position_mixer 不是 depthwise Conv3d："
                        f"{tuple(tensor.shape)}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
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
                        "AttnRes position_mixer spatial kernel "
                        f"不是正方形：{tuple(tensor.shape)}"
                    )

                spatial_kernel_sizes.add(
                    spatial_h
                )

                saw_core_weight = True

            elif attention_projection_pattern.search(
                key_string
            ):
                if tensor.ndim != 2:
                    raise RuntimeError(
                        "AttnRes attention projection 维度异常："
                        f"{key_string} -> {tuple(tensor.shape)}"
                    )

                if tensor.shape[0] != tensor.shape[1]:
                    raise RuntimeError(
                        "AttnRes attention projection 必须为 CxC："
                        f"{tuple(tensor.shape)}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
                )

                saw_core_weight = True

            if time_first_pattern.search(
                key_string
            ):
                if tensor.ndim != 2:
                    raise RuntimeError(
                        "AttnRes time_modulation.1.weight "
                        f"维度异常：{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != model_channels:
                    raise RuntimeError(
                        "AttnRes time_modulation 输入 channel "
                        "与当前模型不一致："
                        f"{tensor.shape[1]} != {model_channels}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
                )

            if time_last_pattern.search(
                key_string
            ):
                if tensor.ndim != 2:
                    raise RuntimeError(
                        "AttnRes time_modulation.3.weight "
                        f"维度异常：{tuple(tensor.shape)}"
                    )

                output_channels = int(
                    tensor.shape[0]
                )

                input_channels = int(
                    tensor.shape[1]
                )

                if output_channels != 4 * input_channels:
                    raise RuntimeError(
                        "AttnRes time_modulation.3.weight "
                        "形状不符合 C -> 4C："
                        f"{tuple(tensor.shape)}"
                    )

                attention_channels.add(
                    input_channels
                )

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
                        "AttnRes memory depthwise.weight "
                        f"维度异常：{tuple(tensor.shape)}"
                    )

                if int(tensor.shape[1]) != 1:
                    raise RuntimeError(
                        "AttnRes memory depthwise.weight "
                        "不是 depthwise Conv3d："
                        f"{tuple(tensor.shape)}"
                    )

                attention_channels.add(
                    int(tensor.shape[0])
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
                        "AttnRes memory spatial kernel 不是正方形："
                        f"{tuple(tensor.shape)}"
                    )

                spatial_kernel_sizes.add(
                    spatial_h
                )

                saw_any_memory_unit_weight = True

        if len(attention_channels) > 1:
            raise RuntimeError(
                "同一个 AttnRes checkpoint 包含多个 "
                "attention_channels："
                f"{sorted(attention_channels)}"
            )

        if attention_channels:
            channel = next(
                iter(attention_channels)
            )

            config.attention_channels_override = (
                channel
            )

            config.attention_ratio = (
                float(channel)
                / float(model_channels)
            )

        if memory_unit_indices:
            config.memory_depth = (
                max(memory_unit_indices) + 1
            )
        elif (
            saw_core_weight
            and not saw_any_memory_unit_weight
        ):
            config.memory_depth = 0

        if len(spatial_kernel_sizes) > 1:
            raise RuntimeError(
                "同一个 AttnRes checkpoint 包含多个 "
                "spatial kernel size："
                f"{sorted(spatial_kernel_sizes)}"
            )

        if spatial_kernel_sizes:
            config.spatial_kernel_size = next(
                iter(spatial_kernel_sizes)
            )

        if len(temporal_kernel_sizes) > 1:
            raise RuntimeError(
                "同一个 AttnRes checkpoint 包含多个 "
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
        原地安装 Hybrid MUDD + AttnRes V3。

        source_keys:
            基础模型或 LoRA 参数键名。

        source_state_dict:
            可选的完整参数字典。若调用方提供，则可从权重形状恢复
            精确 channel/depth/kernel。
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

        if not hasattr(
            model,
            "blocks",
        ):
            raise AttributeError(
                "Anima diffusion_model 不存在 blocks 属性"
            )

        if not hasattr(
            model,
            "model_channels",
        ):
            raise AttributeError(
                "Anima diffusion_model 不存在 model_channels 属性"
            )

        num_blocks = len(
            model.blocks
        )

        if num_blocks <= 0:
            raise RuntimeError(
                "Anima diffusion_model.blocks 为空"
            )

        model_channels = int(
            model.model_channels
        )

        all_source_keys = list(
            source_keys or []
        )

        if source_state_dict:
            all_source_keys.extend(
                source_state_dict.keys()
            )

        mudd_indices = (
            cls._infer_namespace_indices(
                all_source_keys,
                cls.MUDD_NAMESPACE,
            )
        )

        attnres_indices = (
            cls._infer_namespace_indices(
                all_source_keys,
                cls.ATTNRES_NAMESPACE,
            )
        )

        if not mudd_indices:
            mudd_indices = (
                build_every_n_block_indices(
                    num_blocks=num_blocks,
                    stride=(
                        cls.DEFAULT_MUDD_CONFIG
                        .block_stride
                    ),
                    include_last_partial=False,
                )
            )

        if not attnres_indices:
            attnres_indices = (
                build_attnres_block_indices(
                    num_blocks=num_blocks,
                    mudd_block_indices=(
                        mudd_indices
                    ),
                    group_size=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .group_size
                    ),
                    include_partial_group=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .include_partial_group
                    ),
                    placement=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .placement
                    ),
                )
            )

        if not mudd_indices:
            raise RuntimeError(
                "没有可安装的 MUDD block"
            )

        if not attnres_indices:
            raise RuntimeError(
                "没有可安装的 AttnRes block"
            )

        mudd_config = (
            cls._infer_mudd_config(
                source_state_dict,
                model_channels=model_channels,
            )
        )

        attnres_config = (
            cls._infer_attnres_config(
                source_state_dict,
                model_channels=model_channels,
            )
        )

        reference_parameter = (
            _find_model_reference_parameter(
                model
            )
        )

        mudd_set = {
            int(index)
            for index in mudd_indices
        }

        attnres_set = {
            int(index)
            for index in attnres_indices
        }

        for block_index in sorted(
            mudd_set
        ):
            if not (
                0 <= block_index < num_blocks
            ):
                raise ValueError(
                    "无效的 MUDD block index："
                    f"{block_index}，总 blocks={num_blocks}"
                )

            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.MUDD_NAMESPACE,
            ):
                graft = getattr(
                    block,
                    cls.MUDD_NAMESPACE,
                )

                if not isinstance(
                    graft,
                    MUDDFormerGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.MUDD_NAMESPACE}："
                        f"{type(graft).__name__}"
                    )
            else:
                graft = MUDDFormerGraft(
                    model_channels=model_channels,
                    config=mudd_config,
                    block_index=block_index,
                )

                _move_new_module_like_model(
                    graft,
                    reference_parameter,
                )

                block.add_module(
                    cls.MUDD_NAMESPACE,
                    graft,
                )

        for block_index in sorted(
            attnres_set
        ):
            if not (
                0 <= block_index < num_blocks
            ):
                raise ValueError(
                    "无效的 AttnRes block index："
                    f"{block_index}，总 blocks={num_blocks}"
                )

            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.ATTNRES_NAMESPACE,
            ):
                graft = getattr(
                    block,
                    cls.ATTNRES_NAMESPACE,
                )

                if not isinstance(
                    graft,
                    AttnResGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.ATTNRES_NAMESPACE}："
                        f"{type(graft).__name__}"
                    )
            else:
                graft = AttnResGraft(
                    model_channels=model_channels,
                    config=attnres_config,
                    block_index=block_index,
                )

                _move_new_module_like_model(
                    graft,
                    reference_parameter,
                )

                block.add_module(
                    cls.ATTNRES_NAMESPACE,
                    graft,
                )

        # Block 0 必须捕获 patch embedding 后的 shallow feature。
        blocks_to_wrap = (
            mudd_set
            | attnres_set
            | {0}
        )

        for block_index in sorted(
            blocks_to_wrap
        ):
            block = model.blocks[
                block_index
            ]

            _install_hybrid_block_forward(
                block=block,
                model=model,
                block_index=block_index,
                apply_mudd=(
                    block_index in mudd_set
                ),
                apply_attnres=(
                    block_index in attnres_set
                ),
            )

        _install_model_runtime_context_forward(
            model
        )

        object.__setattr__(
            model,
            "mudd_config",
            mudd_config,
        )

        object.__setattr__(
            model,
            "mudd_block_indices",
            sorted(mudd_set),
        )

        object.__setattr__(
            model,
            "attnres_config",
            attnres_config,
        )

        object.__setattr__(
            model,
            "attnres_block_indices",
            sorted(attnres_set),
        )

        object.__setattr__(
            model,
            "graft_config",
            {
                "mudd": mudd_config,
                "attnres": attnres_config,
            },
        )

        object.__setattr__(
            model,
            "graft_block_indices",
            {
                "mudd": sorted(mudd_set),
                "attnres": sorted(attnres_set),
            },
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
            "✅ [SpatialGraftV3] Hybrid MUDD + AttnRes 已安装"
        )

        print(
            "ℹ️ [SpatialGraftV3] MUDD blocks: "
            f"{sorted(mudd_set)}"
        )

        print(
            "ℹ️ [SpatialGraftV3] AttnRes blocks: "
            f"{sorted(attnres_set)}"
        )

        print(
            "ℹ️ [SpatialGraftV3] MUDD 配置: "
            f"memory_channels="
            f"{mudd_config.memory_channels_override}, "
            f"memory_pool={mudd_config.memory_pool}, "
            f"memory_depth={mudd_config.memory_depth}, "
            f"use_detail_branch="
            f"{mudd_config.use_detail_branch}, "
            f"detail_channels="
            f"{mudd_config.detail_channels_override}, "
            f"spatial_kernel="
            f"{mudd_config.spatial_kernel_size}, "
            f"temporal_kernel="
            f"{mudd_config.temporal_kernel_size}"
        )

        print(
            "ℹ️ [SpatialGraftV3] AttnRes 配置: "
            f"attention_channels="
            f"{attnres_config.attention_channels_override}, "
            f"num_heads={attnres_config.num_heads}, "
            f"memory_pool={attnres_config.memory_pool}, "
            f"max_memory_tokens="
            f"{attnres_config.max_memory_tokens}, "
            f"max_temporal_tokens="
            f"{attnres_config.max_temporal_tokens}, "
            f"memory_depth="
            f"{attnres_config.memory_depth}, "
            f"spatial_kernel="
            f"{attnres_config.spatial_kernel_size}, "
            f"temporal_kernel="
            f"{attnres_config.temporal_kernel_size}"
        )

        if reference_parameter is not None:
            print(
                "ℹ️ [SpatialGraftV3] 新增模块参考精度: "
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

    "AttnResGraftRuntimeConfig",
    "AttnResGraft",
    "AttnResMemoryUnit",
    "CompressedAttnResAttention",

    "SimpleRMSNorm",
    "ChannelNorm3d",

    "build_every_n_block_indices",
    "build_attnres_block_indices",
]
