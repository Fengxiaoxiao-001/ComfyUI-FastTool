# coding=utf-8
# Mutation/anima_spatial_graft_v4.py
#
# Anima Spatial Graft V4 runtime mutation.
#
# Image-specialized hybrid grafting:
#
#   1. MUDD-Former
#   2. 2D AttnRes
#   3. Mixture-of-Recursions (MoR)
#
# 新增参数路径：
#
#   blocks.{i}.mudd_graft.*
#   blocks.{i}.spatial_graft.*
#   blocks.{i}.mor_graft.*
#
# 三个 graft 可以安装在同一个原始 Transformer block 上，并且从
# 同一个原始 block 输出并行读取：
#
#   base = original_block(x)
#
#   mudd_delta    = mudd(base)    - base
#   attnres_delta = attnres(base) - base
#   mor_delta     = mor(base)     - base
#
#   output = base + mudd_delta + attnres_delta + mor_delta
#
# 原始 Anima block 不会被包装成新的 nn.Module，原始参数路径保持不变。
#
# MUDD runtime 模块复用同目录 anima_spatial_graft_v2.py，确保 V2/V4
# 的 MUDD 参数结构和数值路径一致。
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import math
import re
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 加载 V2 MUDD runtime
# ============================================================


def _load_v2_runtime_module():
    """
    优先使用正常 package 相对导入。

    某些 AnimaBaker 实现会通过 spec_from_file_location() 独立扫描
    Mutation 文件，此时当前模块可能没有 package。这里提供文件路径
    fallback，保证仍可加载同目录的 V2 runtime。
    """

    try:
        from . import anima_spatial_graft_v2 as module

        return module
    except Exception:
        module_name = "_anima_spatial_graft_v2_runtime_shared"

        existing = sys.modules.get(module_name)

        if existing is not None:
            return existing

        module_path = Path(__file__).with_name(
            "anima_spatial_graft_v2.py"
        )

        if not module_path.is_file():
            raise ImportError(
                "anima_spatial_graft_v4 需要同目录中的 "
                "anima_spatial_graft_v2.py，以复用完全兼容的 "
                "MUDD runtime 模块。缺少文件："
                f"{module_path}"
            )

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(module_path),
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                "无法创建 anima_spatial_graft_v2 runtime "
                f"导入规格：{module_path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )

        # dataclass 等运行时逻辑要求模块在执行前进入 sys.modules。
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(
                module_name,
                None,
            )
            raise

        return module


_v2_runtime = _load_v2_runtime_module()

MUDDGraftRuntimeConfig = (
    _v2_runtime.MUDDGraftRuntimeConfig
)
MUDDFormerGraft = (
    _v2_runtime.MUDDFormerGraft
)
MUDDMemoryUnit = (
    _v2_runtime.MUDDMemoryUnit
)
MUDDDetailUnit = (
    _v2_runtime.MUDDDetailUnit
)
MUDDSimpleRMSNorm = (
    _v2_runtime.SimpleRMSNorm
)
MUDDChannelNorm3d = (
    _v2_runtime.ChannelNorm3d
)

_V2Mutation = (
    _v2_runtime.GraftedAnima
)


# ============================================================
# V4 runtime 配置
# ============================================================


@dataclass
class AttnResRuntimeConfig:
    """
    Image-only 2D AttnRes runtime 配置。

    默认值与 library/anima_grafted4.py 中的
    AttnResGraftConfig 一致。

    attention_channels_override 仅用于从 checkpoint 形状恢复
    精确架构，不会创建额外参数。
    """

    enabled: bool = True

    group_size: int = 3
    include_partial_group: bool = False
    placement: str = "middle"

    attention_ratio: float = 0.0625
    attention_channels_override: Optional[int] = None

    memory_pool: int = 4
    max_memory_tokens: int = 128
    num_heads: int = 4
    memory_depth: int = 1

    spatial_kernel_size: int = 3
    dropout: float = 0.0

    attention_layer_scale_init: float = 0.1
    memory_layer_scale_init: float = 0.1
    branch_scale_init: float = 1.0
    output_init_std: float = 0.0
    memory_update_bias: float = -1.5

    num_prototype_tokens: int = 16

    use_rms_norm: bool = True
    strict_image_only: bool = True


@dataclass
class MoRRuntimeConfig:
    """
    Image-only Mixture-of-Recursions runtime 配置。

    默认值与 library/anima_grafted4.py 中的 MoRGraftConfig 一致。

    channels_override 仅用于从 checkpoint 恢复精确 channel。
    """

    enabled: bool = True

    block_stride: int = 4
    block_offset: int = 2
    include_last_block: bool = False

    channel_ratio: float = 0.0625
    channels_override: Optional[int] = None

    num_heads: int = 4

    max_recursions: int = 3
    max_memory_tokens: int = 128
    memory_pool: int = 4
    memory_depth: int = 1

    spatial_kernel_size: int = 3
    dilation: int = 2
    dropout: float = 0.0

    recurrent_layer_scale_init: float = 0.1
    local_layer_scale_init: float = 0.1
    branch_scale_init: float = 1.0
    output_init_std: float = 0.0
    memory_update_bias: float = -1.5

    num_prototype_tokens: int = 24

    use_text_conditioning: bool = True
    max_text_tokens: int = 128

    use_rms_norm: bool = True
    strict_image_only: bool = True


@dataclass
class ExtraBlockRuntimeConfig:
    """Residual Anima blocks stored outside the original 28-block ModuleList."""

    enabled: bool = False
    num_blocks: int = 0
    insert_after: Tuple[int, ...] = ()
    source_indices: Tuple[int, ...] = ()
    residual_scale_init: float = 0.05


class ExtraAnimaBlock(nn.Module):
    """A cloned base block with a bounded, checkpoint-compatible residual gate."""

    def __init__(
        self,
        block: nn.Module,
        insert_after: int,
        source_index: int,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.block = block
        self.insert_after = int(insert_after)
        self.source_index = int(source_index)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        refined = self.block(
            x_B_T_H_W_D,
            emb_B_T_D,
            *args,
            **kwargs,
        )
        refined_tensor, _ = _extract_primary_tensor(refined)
        if not torch.is_tensor(refined_tensor):
            raise TypeError("Extra Anima block 主输出不是 Tensor")
        if refined_tensor.shape != x_B_T_H_W_D.shape:
            raise ValueError(
                "Extra Anima block 输出形状与输入不一致："
                f"{tuple(refined_tensor.shape)} != {tuple(x_B_T_H_W_D.shape)}"
            )
        scale = torch.tanh(self.residual_scale).to(
            device=refined_tensor.device,
            dtype=refined_tensor.dtype,
        )
        return x_B_T_H_W_D + scale * (
            refined_tensor - x_B_T_H_W_D
        )


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
    查找新增模块应跟随的全局 dtype/device。
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


def _find_block_reference_parameter(
    block: nn.Module,
    fallback: Optional[nn.Parameter],
) -> Optional[nn.Parameter]:
    """
    block swap 场景下优先跟随当前 block 的设备和精度。
    """

    preferred_modules = (
        getattr(block, "self_attn", None),
        getattr(block, "cross_attn", None),
        getattr(block, "mlp", None),
    )

    for module in preferred_modules:
        parameter = _module_first_floating_parameter(
            module
        )

        if parameter is not None:
            return parameter

    parameter = _module_first_floating_parameter(
        block
    )

    return (
        parameter
        if parameter is not None
        else fallback
    )


def _move_module_like_parameter(
    module: nn.Module,
    reference_parameter: Optional[nn.Parameter],
):
    if reference_parameter is None:
        return

    reference_dtype = reference_parameter.dtype
    reference_device = reference_parameter.device

    if not reference_parameter.is_floating_point():
        return

    if reference_device.type == "meta":
        # 不把新建实体参数直接移动成 meta，否则常规 .to() 无法
        # 再恢复实体 storage。这里只同步 dtype。
        module.to(
            dtype=reference_dtype
        )
    else:
        module.to(
            device=reference_device,
            dtype=reference_dtype,
        )


def _require_same_device(
    tensor: torch.Tensor,
    expected_device: torch.device,
    tensor_name: str,
):
    if tensor.device != expected_device:
        raise RuntimeError(
            f"[SpatialGraftV4] {tensor_name} 与 graft 模块不在"
            "同一设备："
            f"tensor={tensor.device}, module={expected_device}。\n"
            "这通常表示 block swap 或 ComfyUI ModelPatcher 没有"
            "把新增 graft 子模块与原 block 一起移动。"
        )


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


def _check_finite(
    tensor: torch.Tensor,
    tensor_name: str,
):
    if not tensor.is_floating_point():
        return

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"[SpatialGraftV4] {tensor_name} 中检测到 NaN/Inf。"
            "已终止推理，避免继续生成黑图或损坏结果。"
        )


def _validate_odd_kernel(
    kernel_size: int,
    name: str,
):
    kernel_size = int(kernel_size)

    if (
        kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            f"{name} 必须是正奇数，实际为 {kernel_size}"
        )


def _safe_interpolate_2d(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """
    bilinear interpolate 后严格恢复输入 dtype/device。

    少数 CPU backend 不支持 FP16 bilinear，此时仅在算子内部临时
    使用 FP32，返回值仍恢复到原精度。
    """

    input_dtype = x.dtype
    input_device = x.device

    try:
        output = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    except RuntimeError:
        if (
            input_device.type == "cpu"
            and input_dtype in (
                torch.float16,
                torch.bfloat16,
            )
        ):
            output = F.interpolate(
                x.float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise

    return output.to(
        device=input_device,
        dtype=input_dtype,
    )


def _safe_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    input_dtype = x.dtype
    input_device = x.device

    try:
        output = F.avg_pool2d(
            x,
            kernel_size=kernel_size,
            stride=kernel_size,
            ceil_mode=True,
            count_include_pad=False,
        )
    except RuntimeError:
        if (
            input_device.type == "cpu"
            and input_dtype in (
                torch.float16,
                torch.bfloat16,
            )
        ):
            output = F.avg_pool2d(
                x.float(),
                kernel_size=kernel_size,
                stride=kernel_size,
                ceil_mode=True,
                count_include_pad=False,
            )
        else:
            raise

    return output.to(
        device=input_device,
        dtype=input_dtype,
    )


def _safe_adaptive_avg_pool2d(
    x: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    input_dtype = x.dtype
    input_device = x.device

    try:
        output = F.adaptive_avg_pool2d(
            x,
            output_size=output_size,
        )
    except RuntimeError:
        if (
            input_device.type == "cpu"
            and input_dtype in (
                torch.float16,
                torch.bfloat16,
            )
        ):
            output = F.adaptive_avg_pool2d(
                x.float(),
                output_size=output_size,
            )
        else:
            raise

    return output.to(
        device=input_device,
        dtype=input_dtype,
    )


# ============================================================
# 通用形状工具
# ============================================================


def _round_channels(
    model_channels: int,
    ratio: float,
    override: Optional[int] = None,
    multiple: int = 32,
    minimum: int = 32,
) -> int:
    model_channels = int(
        model_channels
    )

    if override is not None:
        channels = int(
            override
        )

        if channels <= 0:
            raise ValueError(
                "channel override 必须大于 0"
            )

        if channels > model_channels:
            raise ValueError(
                "channel override 不能超过 model_channels："
                f"{channels} > {model_channels}"
            )

        return channels

    channels = max(
        int(minimum),
        int(
            round(
                model_channels
                * float(ratio)
            )
        ),
    )

    channels = int(
        math.ceil(
            channels / multiple
        ) * multiple
    )

    return min(
        channels,
        model_channels,
    )


def _valid_num_heads(
    channels: int,
    requested_heads: int,
) -> int:
    channels = int(
        channels
    )

    heads = min(
        max(
            1,
            int(requested_heads),
        ),
        channels,
    )

    while (
        heads > 1
        and channels % heads != 0
    ):
        heads -= 1

    return heads


def _ensure_image_tensor(
    x_B_T_H_W_D: torch.Tensor,
    strict: bool,
    module_name: str,
) -> torch.Tensor:
    if not torch.is_tensor(
        x_B_T_H_W_D
    ):
        raise TypeError(
            f"{module_name} 输入必须是 Tensor"
        )

    if x_B_T_H_W_D.ndim != 5:
        raise ValueError(
            f"{module_name} 期望 [B,T,H,W,D]，实际为 "
            f"{tuple(x_B_T_H_W_D.shape)}"
        )

    if (
        strict
        and x_B_T_H_W_D.shape[1] != 1
    ):
        raise ValueError(
            f"{module_name} 是 image-only graft，要求 patch "
            "embedding 后 T=1，实际为 "
            f"{x_B_T_H_W_D.shape[1]}"
        )

    if x_B_T_H_W_D.shape[1] == 1:
        return x_B_T_H_W_D[:, 0]

    # 非严格兼容路径。
    return x_B_T_H_W_D.mean(
        dim=1
    )


def _prepare_image_timestep(
    timestep_embedding_B_T_D: torch.Tensor,
    batch_size: int,
    model_channels: int,
) -> torch.Tensor:
    if not torch.is_tensor(
        timestep_embedding_B_T_D
    ):
        raise TypeError(
            "timestep embedding 必须是 Tensor"
        )

    if timestep_embedding_B_T_D.ndim != 3:
        raise ValueError(
            "timestep embedding 期望 [B,T,D]，实际为 "
            f"{tuple(timestep_embedding_B_T_D.shape)}"
        )

    if (
        timestep_embedding_B_T_D.shape[0]
        != batch_size
    ):
        raise ValueError(
            "timestep embedding batch 不一致"
        )

    if (
        timestep_embedding_B_T_D.shape[-1]
        != model_channels
    ):
        raise ValueError(
            "timestep embedding channel 不一致："
            f"{timestep_embedding_B_T_D.shape[-1]} != "
            f"{model_channels}"
        )

    return timestep_embedding_B_T_D.mean(
        dim=1
    )


def _calculate_spatial_budget(
    height: int,
    width: int,
    max_tokens: int,
) -> Tuple[int, int]:
    height = int(
        height
    )
    width = int(
        width
    )
    max_tokens = max(
        1,
        int(max_tokens),
    )

    if height * width <= max_tokens:
        return height, width

    aspect = (
        float(height)
        / max(float(width), 1.0)
    )

    target_h = max(
        1,
        int(
            math.sqrt(
                max_tokens * aspect
            )
        ),
    )

    target_w = max(
        1,
        max_tokens // target_h,
    )

    target_h = min(
        height,
        target_h,
    )
    target_w = min(
        width,
        target_w,
    )

    while target_h * target_w > max_tokens:
        if (
            target_h >= target_w
            and target_h > 1
        ):
            target_h -= 1
        elif target_w > 1:
            target_w -= 1
        else:
            break

    return (
        max(1, target_h),
        max(1, target_w),
    )


def _pool_image_memory(
    x: torch.Tensor,
    initial_pool: int,
    max_tokens: int,
) -> torch.Tensor:
    input_dtype = x.dtype
    input_device = x.device

    initial_pool = int(
        initial_pool
    )

    if initial_pool <= 0:
        raise ValueError(
            "memory_pool 必须大于 0"
        )

    if initial_pool > 1:
        x = _safe_avg_pool2d(
            x,
            initial_pool,
        )

    height = int(
        x.shape[-2]
    )
    width = int(
        x.shape[-1]
    )

    target_h, target_w = (
        _calculate_spatial_budget(
            height,
            width,
            max_tokens,
        )
    )

    if (
        target_h,
        target_w,
    ) != (
        height,
        width,
    ):
        x = _safe_adaptive_avg_pool2d(
            x,
            (
                target_h,
                target_w,
            ),
        )

    return x.to(
        device=input_device,
        dtype=input_dtype,
    )


def _resize_image_memory(
    memory: Optional[torch.Tensor],
    reference: torch.Tensor,
    fallback: torch.Tensor,
    expected_channels: int,
    module_name: str,
) -> torch.Tensor:
    if memory is None:
        return fallback

    if not torch.is_tensor(
        memory
    ):
        raise TypeError(
            f"{module_name} previous_memory 必须是 Tensor 或 None"
        )

    if memory.ndim != 4:
        raise ValueError(
            f"{module_name} previous_memory 期望 [B,C,H,W]，"
            f"实际为 {tuple(memory.shape)}"
        )

    if memory.shape[0] != reference.shape[0]:
        raise ValueError(
            f"{module_name} persistent memory batch 发生变化："
            f"{memory.shape[0]} != {reference.shape[0]}"
        )

    if memory.shape[1] != expected_channels:
        raise ValueError(
            f"{module_name} persistent memory channel 不一致："
            f"{memory.shape[1]} != {expected_channels}"
        )

    memory = memory.to(
        device=reference.device,
        dtype=reference.dtype,
    )

    if memory.shape[-2:] != reference.shape[-2:]:
        memory = _safe_interpolate_2d(
            memory,
            (
                int(reference.shape[-2]),
                int(reference.shape[-1]),
            ),
        )

    return memory


# ============================================================
# Normalization
# ============================================================


class SimpleRMSNorm(nn.Module):
    """
    无 affine RMSNorm。

    内部 FP32 计算 RMS，输出恢复输入 dtype/device。
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.dim = int(
            dim
        )
        self.eps = float(
            eps
        )

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
            output = (
                output
                * self.weight.to(
                    device=input_device,
                    dtype=input_dtype,
                )
            )

            output = output.to(
                device=input_device,
                dtype=input_dtype,
            )

        return output


class ChannelNorm2d(nn.Module):
    """
    对 B,C,H,W 的 C 维执行 RMSNorm 或 LayerNorm。

    参数路径与训练架构一致：

        norm.weight
        norm.bias
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        self.channels = int(
            channels
        )
        self.eps = float(
            eps
        )
        self.use_rms_norm = bool(
            use_rms_norm
        )

        self.weight = nn.Parameter(
            torch.ones(
                self.channels,
                dtype=torch.float32,
            )
        )

        if self.use_rms_norm:
            self.register_parameter(
                "bias",
                None,
            )
        else:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.channels,
                    dtype=torch.float32,
                )
            )

    def reset_parameters(self):
        nn.init.ones_(
            self.weight
        )

        if self.bias is not None:
            nn.init.zeros_(
                self.bias
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ChannelNorm2d 期望 [B,C,H,W]，实际为 "
                f"{tuple(x.shape)}"
            )

        input_dtype = x.dtype
        input_device = x.device

        x_float = x.float()

        if self.use_rms_norm:
            variance = x_float.square().mean(
                dim=1,
                keepdim=True,
            )

            output = (
                x_float
                * torch.rsqrt(
                    variance + self.eps
                )
            )
        else:
            mean = x_float.mean(
                dim=1,
                keepdim=True,
            )

            centered = (
                x_float - mean
            )

            variance = centered.square().mean(
                dim=1,
                keepdim=True,
            )

            output = (
                centered
                * torch.rsqrt(
                    variance + self.eps
                )
            )

        weight = self.weight.float().reshape(
            1,
            -1,
            1,
            1,
        )

        output = output * weight

        if self.bias is not None:
            output = (
                output
                + self.bias.float().reshape(
                    1,
                    -1,
                    1,
                    1,
                )
            )

        return output.to(
            device=input_device,
            dtype=input_dtype,
        )


# ============================================================
# 2D memory unit
# ============================================================


class ImageMemoryUnit(nn.Module):
    """
    与训练架构兼容的 2D image memory unit。
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

        channels = int(
            channels
        )

        _validate_odd_kernel(
            spatial_kernel_size,
            "spatial_kernel_size",
        )

        self.channels = channels

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=int(
                spatial_kernel_size
            ),
            padding=int(
                spatial_kernel_size
            ) // 2,
            groups=channels,
            bias=False,
        )

        self.norm = ChannelNorm2d(
            channels,
            eps=1e-6,
            use_rms_norm=use_rms_norm,
        )

        self.pointwise_in = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=1,
            bias=False,
        )

        self.pointwise_out = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = (
            nn.Dropout2d(
                float(dropout)
            )
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

        self.norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = self.depthwise.weight.dtype
        compute_device = self.depthwise.weight.device

        _require_same_device(
            x,
            compute_device,
            "ImageMemoryUnit 输入",
        )

        x = _cast_tensor(
            x,
            compute_dtype,
            compute_device,
        )

        residual = x

        branch = self.depthwise(
            x
        )

        branch = self.norm(
            branch
        )

        branch = _cast_tensor(
            branch,
            compute_dtype,
            compute_device,
        )

        branch = self.pointwise_in(
            branch
        )

        branch = F.gelu(
            branch,
            approximate="tanh",
        )

        branch = self.pointwise_out(
            branch
        )

        branch = self.dropout(
            branch
        )

        branch = _cast_tensor(
            branch,
            compute_dtype,
            compute_device,
        )

        scale = self.layer_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
        )

        output = (
            residual
            + scale * branch
        )

        return _cast_tensor(
            output,
            compute_dtype,
            compute_device,
        )


# ============================================================
# Compressed image attention
# ============================================================


class CompressedImageAttention(nn.Module):
    """
    QK-normalized SDPA attention。

    query:
        [B,Lq,C]

    context:
        [B,Lk,C]

    context_mask:
        可选 [B,Lk] bool，True 表示允许访问该 key。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        channels = int(
            channels
        )
        num_heads = int(
            num_heads
        )

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} 不能被 "
                f"num_heads={num_heads} 整除"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = (
            channels // num_heads
        )
        self.dropout = float(
            dropout
        )

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
        std = (
            1.0
            / math.sqrt(
                self.channels
            )
        )

        for layer in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
        ):
            nn.init.trunc_normal_(
                layer.weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        compute_dtype = self.q_proj.weight.dtype
        compute_device = self.q_proj.weight.device

        _require_same_device(
            query,
            compute_device,
            "CompressedImageAttention query",
        )

        _require_same_device(
            context,
            compute_device,
            "CompressedImageAttention context",
        )

        query = _cast_tensor(
            query,
            compute_dtype,
            compute_device,
        )

        context = _cast_tensor(
            context,
            compute_dtype,
            compute_device,
        )

        if query.ndim != 3:
            raise ValueError(
                "CompressedImageAttention query 期望 [B,L,C]"
            )

        if context.ndim != 3:
            raise ValueError(
                "CompressedImageAttention context 期望 [B,L,C]"
            )

        (
            batch_size,
            query_length,
            channels,
        ) = query.shape

        (
            context_batch,
            context_length,
            context_channels,
        ) = context.shape

        if (
            batch_size != context_batch
            or channels != context_channels
            or channels != self.channels
        ):
            raise ValueError(
                "CompressedImageAttention query/context 形状不兼容："
                f"{tuple(query.shape)} vs {tuple(context.shape)}"
            )

        q = self.q_proj(
            query
        ).reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )

        k = self.k_proj(
            context
        ).reshape(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        )

        v = self.v_proj(
            context
        ).reshape(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        )

        q = self.q_norm(
            q
        ).transpose(
            1,
            2,
        )

        k = self.k_norm(
            k
        ).transpose(
            1,
            2,
        )

        v = v.transpose(
            1,
            2,
        )

        q = _cast_tensor(
            q,
            compute_dtype,
            compute_device,
        )
        k = _cast_tensor(
            k,
            compute_dtype,
            compute_device,
        )
        v = _cast_tensor(
            v,
            compute_dtype,
            compute_device,
        )

        attn_mask = None

        if context_mask is not None:
            if context_mask.ndim != 2:
                raise ValueError(
                    "context_mask 必须是 [B,Lk]"
                )

            if (
                context_mask.shape[0]
                != batch_size
                or context_mask.shape[1]
                != context_length
            ):
                raise ValueError(
                    "CompressedImageAttention mask 形状不匹配："
                    f"{tuple(context_mask.shape)}，期望 "
                    f"({batch_size}, {context_length})"
                )

            attn_mask = context_mask.to(
                device=compute_device,
                dtype=torch.bool,
            )[:, None, None, :]

        dropout_p = (
            self.dropout
            if self.training
            else 0.0
        )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        output = _cast_tensor(
            output,
            compute_dtype,
            compute_device,
        )

        output = output.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            query_length,
            channels,
        )

        output = self.o_proj(
            output
        )

        return _cast_tensor(
            output,
            compute_dtype,
            compute_device,
        )


# ============================================================
# Image-only AttnRes
# ============================================================


class ImageAttnResGraft(nn.Module):
    """
    低分辨率 residual attention memory。

    参数结构与 library/anima_grafted4.py 中的
    ImageAttnResGraft 一致。
    """

    def __init__(
        self,
        model_channels: int,
        config: AttnResRuntimeConfig,
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

        _validate_odd_kernel(
            config.spatial_kernel_size,
            "AttnRes spatial_kernel_size",
        )

        if int(config.memory_depth) < 0:
            raise ValueError(
                "AttnRes memory_depth 不能小于 0"
            )

        if int(config.memory_pool) <= 0:
            raise ValueError(
                "AttnRes memory_pool 必须大于 0"
            )

        channels = _round_channels(
            model_channels=self.model_channels,
            ratio=config.attention_ratio,
            override=(
                config.attention_channels_override
            ),
        )

        heads = _valid_num_heads(
            channels,
            config.num_heads,
        )

        self.attention_channels = int(
            channels
        )
        self.num_heads = int(
            heads
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

        self.current_proj = nn.Conv2d(
            self.model_channels,
            self.attention_channels,
            kernel_size=1,
            bias=False,
        )

        self.shallow_proj = nn.Conv2d(
            self.model_channels,
            self.attention_channels,
            kernel_size=1,
            bias=False,
        )

        self.position_mixer = nn.Conv2d(
            self.attention_channels,
            self.attention_channels,
            kernel_size=int(
                config.spatial_kernel_size
            ),
            padding=int(
                config.spatial_kernel_size
            ) // 2,
            groups=self.attention_channels,
            bias=False,
        )

        self.attention = CompressedImageAttention(
            channels=self.attention_channels,
            num_heads=self.num_heads,
            dropout=config.dropout,
        )

        self.attention_layer_scale = nn.Parameter(
            torch.full(
                (
                    self.attention_channels,
                ),
                float(
                    config.attention_layer_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.memory_units = nn.ModuleList(
            [
                ImageMemoryUnit(
                    channels=self.attention_channels,
                    spatial_kernel_size=(
                        config.spatial_kernel_size
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

        self.prototype_tokens = nn.Parameter(
            torch.empty(
                max(
                    0,
                    int(
                        config.num_prototype_tokens
                    ),
                ),
                self.attention_channels,
            )
        )

        # update, output, shallow, previous
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

        self.output_proj = nn.Conv2d(
            self.attention_channels,
            self.model_channels,
            kernel_size=1,
            bias=False,
        )

        self.branch_scale = nn.Parameter(
            torch.tensor(
                float(
                    config.branch_scale_init
                ),
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

        if self.prototype_tokens.numel() > 0:
            nn.init.normal_(
                self.prototype_tokens,
                mean=0.0,
                std=0.02,
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
            self.time_modulation[3].bias[
                :self.attention_channels
            ].fill_(
                float(
                    self.config.memory_update_bias
                )
            )

        # Function-preserving initialization。
        nn.init.zeros_(
            self.output_proj.weight
        )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        shallow_B_T_H_W_D: torch.Tensor,
        previous_memory: Optional[torch.Tensor],
        timestep_embedding_B_T_D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            compute_dtype,
            compute_device,
        )

        shallow_B_T_H_W_D = _cast_tensor(
            shallow_B_T_H_W_D,
            compute_dtype,
            compute_device,
        )

        timestep_embedding_B_T_D = _cast_tensor(
            timestep_embedding_B_T_D,
            compute_dtype,
            compute_device,
        )

        residual_input = x_B_T_H_W_D

        current = _ensure_image_tensor(
            x_B_T_H_W_D,
            self.config.strict_image_only,
            "ImageAttnResGraft",
        )

        shallow = _ensure_image_tensor(
            shallow_B_T_H_W_D,
            self.config.strict_image_only,
            "ImageAttnResGraft shallow input",
        )

        if current.shape != shallow.shape:
            raise ValueError(
                "AttnRes current/shallow 形状不一致："
                f"{tuple(current.shape)} vs "
                f"{tuple(shallow.shape)}"
            )

        (
            batch_size,
            height,
            width,
            channels,
        ) = current.shape

        if channels != self.model_channels:
            raise ValueError(
                "AttnRes 输入 channel 不一致："
                f"{channels} != {self.model_channels}"
            )

        timestep = _prepare_image_timestep(
            timestep_embedding_B_T_D,
            batch_size,
            self.model_channels,
        )

        timestep = _cast_tensor(
            timestep,
            compute_dtype,
            compute_device,
        )

        current_norm = self.input_norm(
            current
        )

        shallow_norm = self.shallow_norm(
            shallow
        )

        current_norm = _cast_tensor(
            current_norm,
            compute_dtype,
            compute_device,
        )

        shallow_norm = _cast_tensor(
            shallow_norm,
            compute_dtype,
            compute_device,
        )

        current_full = self.current_proj(
            current_norm.permute(
                0,
                3,
                1,
                2,
            ).contiguous()
        )

        shallow_full = self.shallow_proj(
            shallow_norm.permute(
                0,
                3,
                1,
                2,
            ).contiguous()
        )

        current_full = _cast_tensor(
            current_full,
            compute_dtype,
            compute_device,
        )

        shallow_full = _cast_tensor(
            shallow_full,
            compute_dtype,
            compute_device,
        )

        current_memory = _pool_image_memory(
            current_full,
            self.config.memory_pool,
            self.config.max_memory_tokens,
        )

        shallow_memory = _pool_image_memory(
            shallow_full,
            self.config.memory_pool,
            self.config.max_memory_tokens,
        )

        if (
            shallow_memory.shape[-2:]
            != current_memory.shape[-2:]
        ):
            shallow_memory = _safe_interpolate_2d(
                shallow_memory,
                (
                    int(current_memory.shape[-2]),
                    int(current_memory.shape[-1]),
                ),
            )

        previous_memory = _resize_image_memory(
            memory=previous_memory,
            reference=current_memory,
            fallback=shallow_memory,
            expected_channels=self.attention_channels,
            module_name="AttnRes",
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
            compute_dtype,
            compute_device,
        )

        shallow_positioned = _cast_tensor(
            shallow_positioned,
            compute_dtype,
            compute_device,
        )

        previous_positioned = _cast_tensor(
            previous_positioned,
            compute_dtype,
            compute_device,
        )

        modulation = self.time_modulation(
            timestep
        )

        modulation = _cast_tensor(
            modulation,
            compute_dtype,
            compute_device,
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

        update_strength = torch.sigmoid(
            update_logits
        )[:, :, None, None]

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
            + half
            * torch.tanh(
                output_gate
            )
        )[:, :, None, None]

        shallow_strength = (
            one
            + half
            * torch.tanh(
                shallow_gate
            )
        )[:, :, None, None]

        previous_strength = (
            one
            + half
            * torch.tanh(
                previous_gate
            )
        )[:, :, None, None]

        update_strength = _cast_tensor(
            update_strength,
            compute_dtype,
            compute_device,
        )

        output_strength = _cast_tensor(
            output_strength,
            compute_dtype,
            compute_device,
        )

        shallow_strength = _cast_tensor(
            shallow_strength,
            compute_dtype,
            compute_device,
        )

        previous_strength = _cast_tensor(
            previous_strength,
            compute_dtype,
            compute_device,
        )

        current_tokens = current_positioned.permute(
            0,
            2,
            3,
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
            1,
        ).reshape(
            batch_size,
            -1,
            self.attention_channels,
        )

        context_parts = [
            current_tokens,
            shallow_tokens,
            previous_tokens,
        ]

        if self.prototype_tokens.numel() > 0:
            prototype_tokens = (
                self.prototype_tokens[
                    None
                ].expand(
                    batch_size,
                    -1,
                    -1,
                ).to(
                    device=compute_device,
                    dtype=compute_dtype,
                )
            )

            context_parts.append(
                prototype_tokens
            )

        context_tokens = torch.cat(
            context_parts,
            dim=1,
        )

        context_tokens = _cast_tensor(
            context_tokens,
            compute_dtype,
            compute_device,
        )

        attended_tokens = self.attention(
            current_tokens,
            context_tokens,
        )

        memory_height = int(
            current_memory.shape[-2]
        )
        memory_width = int(
            current_memory.shape[-1]
        )

        attended_memory = attended_tokens.reshape(
            batch_size,
            memory_height,
            memory_width,
            self.attention_channels,
        ).permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        attended_memory = _cast_tensor(
            attended_memory,
            compute_dtype,
            compute_device,
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
            )
        )

        candidate_memory = (
            current_memory
            + attention_scale
            * attended_memory
        )

        candidate_memory = _cast_tensor(
            candidate_memory,
            compute_dtype,
            compute_device,
        )

        for unit in self.memory_units:
            candidate_memory = unit(
                candidate_memory
            )

        updated_memory = (
            previous_memory
            + update_strength
            * (
                candidate_memory
                - previous_memory
            )
        )

        updated_memory = _cast_tensor(
            updated_memory,
            compute_dtype,
            compute_device,
        )

        visible_memory = (
            updated_memory
            * output_strength
        )

        visible_memory = _safe_interpolate_2d(
            visible_memory,
            (
                int(height),
                int(width),
            ),
        )

        residual = self.output_proj(
            visible_memory
        )

        residual = _cast_tensor(
            residual,
            compute_dtype,
            compute_device,
        )

        residual = residual.permute(
            0,
            2,
            3,
            1,
        )[:, None].contiguous()

        branch_scale = self.branch_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        )

        output = (
            residual_input
            + branch_scale
            * residual
        )

        output = _cast_tensor(
            output,
            compute_dtype,
            compute_device,
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


# 兼容训练架构的公开名称。
AttnResGraft = ImageAttnResGraft


# ============================================================
# MoR shared recursive cell
# ============================================================


class MoRSharedCell(nn.Module):
    """
    权重共享的 image refinement cell。

    同一个实例被重复调用 max_recursions 次。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        spatial_kernel_size: int,
        dilation: int,
        dropout: float,
        recurrent_layer_scale_init: float,
        local_layer_scale_init: float,
        use_rms_norm: bool,
    ):
        super().__init__()

        channels = int(
            channels
        )
        dilation = max(
            1,
            int(dilation),
        )

        _validate_odd_kernel(
            spatial_kernel_size,
            "MoR spatial_kernel_size",
        )

        self.channels = channels

        self.token_norm = SimpleRMSNorm(
            channels,
            eps=1e-6,
            elementwise_affine=False,
        )

        self.context_norm = SimpleRMSNorm(
            channels,
            eps=1e-6,
            elementwise_affine=False,
        )

        self.attention = CompressedImageAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_scale = nn.Parameter(
            torch.full(
                (channels,),
                float(
                    recurrent_layer_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.local_norm = ChannelNorm2d(
            channels,
            eps=1e-6,
            use_rms_norm=use_rms_norm,
        )

        self.local_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=int(
                spatial_kernel_size
            ),
            padding=int(
                spatial_kernel_size
            ) // 2,
            groups=channels,
            bias=False,
        )

        self.dilated_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=int(
                spatial_kernel_size
            ),
            dilation=dilation,
            padding=(
                dilation
                * (
                    int(
                        spatial_kernel_size
                    ) // 2
                )
            ),
            groups=channels,
            bias=False,
        )

        # GEGLU-like local FFN。
        self.ffn_in = nn.Conv2d(
            channels,
            channels * 4,
            kernel_size=1,
            bias=False,
        )

        self.ffn_out = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = (
            nn.Dropout2d(
                float(dropout)
            )
            if dropout > 0.0
            else nn.Identity()
        )

        self.local_scale = nn.Parameter(
            torch.full(
                (channels,),
                float(
                    local_layer_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.attention.reset_parameters()

        nn.init.kaiming_uniform_(
            self.local_depthwise.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.dilated_depthwise.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.ffn_in.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.ffn_out.weight,
            a=math.sqrt(5),
        )

        self.local_norm.reset_parameters()

    def forward(
        self,
        memory: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        compute_dtype = (
            self.local_depthwise.weight.dtype
        )
        compute_device = (
            self.local_depthwise.weight.device
        )

        _require_same_device(
            memory,
            compute_device,
            "MoRSharedCell memory",
        )

        _require_same_device(
            context_tokens,
            compute_device,
            "MoRSharedCell context",
        )

        memory = _cast_tensor(
            memory,
            compute_dtype,
            compute_device,
        )

        context_tokens = _cast_tensor(
            context_tokens,
            compute_dtype,
            compute_device,
        )

        if memory.ndim != 4:
            raise ValueError(
                "MoRSharedCell memory 期望 [B,C,H,W]"
            )

        (
            batch_size,
            channels,
            height,
            width,
        ) = memory.shape

        if channels != self.channels:
            raise ValueError(
                "MoRSharedCell channel 不一致："
                f"{channels} != {self.channels}"
            )

        tokens = memory.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            batch_size,
            height * width,
            channels,
        )

        query = self.token_norm(
            tokens
        )

        context = self.context_norm(
            context_tokens
        )

        query = _cast_tensor(
            query,
            compute_dtype,
            compute_device,
        )

        context = _cast_tensor(
            context,
            compute_dtype,
            compute_device,
        )

        attended = self.attention(
            query,
            context,
            context_mask=context_mask,
        )

        attention_scale = (
            self.attention_scale.to(
                device=compute_device,
                dtype=compute_dtype,
            ).reshape(
                1,
                1,
                -1,
            )
        )

        tokens = (
            tokens
            + attention_scale
            * attended
        )

        tokens = _cast_tensor(
            tokens,
            compute_dtype,
            compute_device,
        )

        memory = tokens.reshape(
            batch_size,
            height,
            width,
            channels,
        ).permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        local = self.local_norm(
            memory
        )

        local = _cast_tensor(
            local,
            compute_dtype,
            compute_device,
        )

        local = (
            self.local_depthwise(
                local
            )
            + self.dilated_depthwise(
                local
            )
        )

        local = _cast_tensor(
            local,
            compute_dtype,
            compute_device,
        )

        value, gate = self.ffn_in(
            local
        ).chunk(
            2,
            dim=1,
        )

        local = (
            value
            * F.gelu(
                gate,
                approximate="tanh",
            )
        )

        local = self.ffn_out(
            local
        )

        local = self.dropout(
            local
        )

        local = _cast_tensor(
            local,
            compute_dtype,
            compute_device,
        )

        local_scale = self.local_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
        )

        output = (
            memory
            + local_scale
            * local
        )

        return _cast_tensor(
            output,
            compute_dtype,
            compute_device,
        )


# ============================================================
# Mixture-of-Recursions graft
# ============================================================


class MoRGraft(nn.Module):
    """
    Image-focused Mixture-of-Recursions graft。

    参数结构与 library/anima_grafted4.py 一致。
    """

    def __init__(
        self,
        model_channels: int,
        context_dim: int,
        config: MoRRuntimeConfig,
        block_index: int,
    ):
        super().__init__()

        if int(config.max_recursions) <= 0:
            raise ValueError(
                "MoR max_recursions 必须大于 0"
            )

        if int(config.memory_depth) < 0:
            raise ValueError(
                "MoR memory_depth 不能小于 0"
            )

        if int(config.memory_pool) <= 0:
            raise ValueError(
                "MoR memory_pool 必须大于 0"
            )

        _validate_odd_kernel(
            config.spatial_kernel_size,
            "MoR spatial_kernel_size",
        )

        self.model_channels = int(
            model_channels
        )
        self.context_dim = int(
            context_dim
        )
        self.block_index = int(
            block_index
        )
        self.config = copy.deepcopy(
            config
        )

        channels = _round_channels(
            model_channels=self.model_channels,
            ratio=config.channel_ratio,
            override=config.channels_override,
        )

        heads = _valid_num_heads(
            channels,
            config.num_heads,
        )

        self.channels = int(
            channels
        )
        self.num_heads = int(
            heads
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

        self.current_proj = nn.Conv2d(
            self.model_channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )

        self.shallow_proj = nn.Conv2d(
            self.model_channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )

        self.position_mixer = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=int(
                config.spatial_kernel_size
            ),
            padding=int(
                config.spatial_kernel_size
            ) // 2,
            groups=self.channels,
            bias=False,
        )

        if config.use_text_conditioning:
            self.text_norm = SimpleRMSNorm(
                self.context_dim,
                eps=1e-6,
                elementwise_affine=False,
            )

            self.text_proj = nn.Linear(
                self.context_dim,
                self.channels,
                bias=False,
            )
        else:
            self.text_norm = nn.Identity()
            self.text_proj = None

        self.prototype_tokens = nn.Parameter(
            torch.empty(
                max(
                    0,
                    int(
                        config.num_prototype_tokens
                    ),
                ),
                self.channels,
            )
        )

        self.shared_cell = MoRSharedCell(
            channels=self.channels,
            num_heads=self.num_heads,
            spatial_kernel_size=(
                config.spatial_kernel_size
            ),
            dilation=max(
                1,
                int(config.dilation),
            ),
            dropout=config.dropout,
            recurrent_layer_scale_init=(
                config.recurrent_layer_scale_init
            ),
            local_layer_scale_init=(
                config.local_layer_scale_init
            ),
            use_rms_norm=config.use_rms_norm,
        )

        self.post_memory_units = nn.ModuleList(
            [
                ImageMemoryUnit(
                    channels=self.channels,
                    spatial_kernel_size=(
                        config.spatial_kernel_size
                    ),
                    dropout=config.dropout,
                    layer_scale_init=(
                        config.local_layer_scale_init
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

        self.time_proj = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.model_channels,
                self.channels,
                bias=False,
            ),
        )

        self.router = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.channels,
                self.channels,
                bias=False,
            ),

            nn.SiLU(),

            nn.Linear(
                self.channels,
                int(config.max_recursions),
                bias=True,
            ),
        )

        self.gate_modulation = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.channels,
                2 * self.channels,
                bias=True,
            ),
        )

        self.detail_norm = ChannelNorm2d(
            self.channels,
            eps=1e-6,
            use_rms_norm=config.use_rms_norm,
        )

        self.detail_depthwise_3 = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False,
        )

        self.detail_depthwise_dilated = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            dilation=2,
            padding=2,
            groups=self.channels,
            bias=False,
        )

        self.detail_proj = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )

        self.detail_scale = nn.Parameter(
            torch.full(
                (self.channels,),
                float(
                    config.local_layer_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.output_proj = nn.Conv2d(
            self.channels,
            self.model_channels,
            kernel_size=1,
            bias=False,
        )

        self.branch_scale = nn.Parameter(
            torch.tensor(
                float(
                    config.branch_scale_init
                ),
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

        if self.text_proj is not None:
            std = (
                1.0
                / math.sqrt(
                    self.context_dim
                )
            )

            nn.init.trunc_normal_(
                self.text_proj.weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

        if self.prototype_tokens.numel() > 0:
            nn.init.normal_(
                self.prototype_tokens,
                mean=0.0,
                std=0.02,
            )

        self.shared_cell.reset_parameters()

        for unit in self.post_memory_units:
            unit.reset_parameters()

        nn.init.normal_(
            self.time_proj[1].weight,
            mean=0.0,
            std=(
                1.0
                / math.sqrt(
                    self.model_channels
                )
            ),
        )

        nn.init.normal_(
            self.router[1].weight,
            mean=0.0,
            std=(
                1.0
                / math.sqrt(
                    self.channels
                )
            ),
        )

        # 初始 recursion mixture 为均匀分布。
        nn.init.zeros_(
            self.router[3].weight
        )
        nn.init.zeros_(
            self.router[3].bias
        )

        nn.init.zeros_(
            self.gate_modulation[1].weight
        )
        nn.init.zeros_(
            self.gate_modulation[1].bias
        )

        with torch.no_grad():
            self.gate_modulation[1].bias[
                :self.channels
            ].fill_(
                float(
                    self.config.memory_update_bias
                )
            )

        nn.init.kaiming_uniform_(
            self.detail_depthwise_3.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.detail_depthwise_dilated.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.detail_proj.weight,
            a=math.sqrt(5),
        )

        self.detail_norm.reset_parameters()

        # Function-preserving initialization。
        nn.init.zeros_(
            self.output_proj.weight
        )

    def _prepare_text(
        self,
        text_context: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        zero_summary = torch.zeros(
            batch_size,
            self.channels,
            device=device,
            dtype=dtype,
        )

        if (
            self.text_proj is None
            or text_context is None
        ):
            return (
                None,
                None,
                zero_summary,
            )

        if not torch.is_tensor(
            text_context
        ):
            raise TypeError(
                "MoR text context 必须是 Tensor 或 None"
            )

        if text_context.ndim != 3:
            raise ValueError(
                "MoR text context 期望 [B,N,D]，实际为 "
                f"{tuple(text_context.shape)}"
            )

        if text_context.shape[0] != batch_size:
            raise ValueError(
                "MoR text context batch 不一致"
            )

        if text_context.shape[-1] != self.context_dim:
            raise ValueError(
                "MoR text context channel 不一致："
                f"{text_context.shape[-1]} != "
                f"{self.context_dim}"
            )

        max_tokens = max(
            1,
            int(
                self.config.max_text_tokens
            ),
        )

        text_context = text_context[
            :,
            :max_tokens,
        ].to(
            device=device,
            dtype=dtype,
        )

        text_normalized = self.text_norm(
            text_context
        )

        text_normalized = _cast_tensor(
            text_normalized,
            self.text_proj.weight.dtype,
            self.text_proj.weight.device,
        )

        text = self.text_proj(
            text_normalized
        )

        text = _cast_tensor(
            text,
            dtype,
            device,
        )

        mask = None

        if text_attention_mask is not None:
            if torch.is_tensor(
                text_attention_mask
            ):
                mask = text_attention_mask

                # [B,1,1,N] / [B,1,N] -> [B,N]
                while (
                    mask.ndim > 2
                    and mask.shape[1] == 1
                ):
                    mask = mask.squeeze(
                        1
                    )

                if mask.ndim == 1:
                    mask = mask.unsqueeze(
                        0
                    )

                if (
                    mask.ndim == 2
                    and mask.shape[0]
                    == batch_size
                    and mask.shape[1]
                    >= text.shape[1]
                ):
                    mask = mask[
                        :,
                        :text.shape[1],
                    ].to(
                        device=device,
                        dtype=torch.bool,
                    )
                else:
                    mask = None

        if mask is None:
            text_summary = text.mean(
                dim=1
            )
        else:
            weights = mask.to(
                dtype=dtype
            ).unsqueeze(
                -1
            )

            denominator = weights.sum(
                dim=1
            ).clamp_min(
                torch.tensor(
                    1.0,
                    device=device,
                    dtype=dtype,
                )
            )

            text_summary = (
                (
                    text * weights
                ).sum(
                    dim=1
                )
                / denominator
            )

        text_summary = _cast_tensor(
            text_summary,
            dtype,
            device,
        )

        return (
            text,
            mask,
            text_summary,
        )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        shallow_B_T_H_W_D: torch.Tensor,
        previous_memory: Optional[torch.Tensor],
        timestep_embedding_B_T_D: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = self.current_proj.weight.dtype
        compute_device = self.current_proj.weight.device

        _require_same_device(
            x_B_T_H_W_D,
            compute_device,
            "MoR 当前特征",
        )

        _require_same_device(
            shallow_B_T_H_W_D,
            compute_device,
            "MoR shallow feature",
        )

        _require_same_device(
            timestep_embedding_B_T_D,
            compute_device,
            "MoR timestep embedding",
        )

        x_B_T_H_W_D = _cast_tensor(
            x_B_T_H_W_D,
            compute_dtype,
            compute_device,
        )

        shallow_B_T_H_W_D = _cast_tensor(
            shallow_B_T_H_W_D,
            compute_dtype,
            compute_device,
        )

        timestep_embedding_B_T_D = _cast_tensor(
            timestep_embedding_B_T_D,
            compute_dtype,
            compute_device,
        )

        residual_input = x_B_T_H_W_D

        current = _ensure_image_tensor(
            x_B_T_H_W_D,
            self.config.strict_image_only,
            "MoRGraft",
        )

        shallow = _ensure_image_tensor(
            shallow_B_T_H_W_D,
            self.config.strict_image_only,
            "MoRGraft shallow input",
        )

        if current.shape != shallow.shape:
            raise ValueError(
                "MoR current/shallow 形状不一致："
                f"{tuple(current.shape)} vs "
                f"{tuple(shallow.shape)}"
            )

        (
            batch_size,
            height,
            width,
            channels,
        ) = current.shape

        if channels != self.model_channels:
            raise ValueError(
                "MoR 输入 channel 不一致："
                f"{channels} != {self.model_channels}"
            )

        timestep = _prepare_image_timestep(
            timestep_embedding_B_T_D,
            batch_size,
            self.model_channels,
        )

        timestep = _cast_tensor(
            timestep,
            compute_dtype,
            compute_device,
        )

        timestep_condition = self.time_proj(
            timestep
        )

        timestep_condition = _cast_tensor(
            timestep_condition,
            compute_dtype,
            compute_device,
        )

        current_norm = self.input_norm(
            current
        )

        shallow_norm = self.shallow_norm(
            shallow
        )

        current_norm = _cast_tensor(
            current_norm,
            compute_dtype,
            compute_device,
        )

        shallow_norm = _cast_tensor(
            shallow_norm,
            compute_dtype,
            compute_device,
        )

        current_full = self.current_proj(
            current_norm.permute(
                0,
                3,
                1,
                2,
            ).contiguous()
        )

        shallow_full = self.shallow_proj(
            shallow_norm.permute(
                0,
                3,
                1,
                2,
            ).contiguous()
        )

        current_full = _cast_tensor(
            current_full,
            compute_dtype,
            compute_device,
        )

        shallow_full = _cast_tensor(
            shallow_full,
            compute_dtype,
            compute_device,
        )

        current_memory = _pool_image_memory(
            current_full,
            self.config.memory_pool,
            self.config.max_memory_tokens,
        )

        shallow_memory = _pool_image_memory(
            shallow_full,
            self.config.memory_pool,
            self.config.max_memory_tokens,
        )

        if (
            shallow_memory.shape[-2:]
            != current_memory.shape[-2:]
        ):
            shallow_memory = _safe_interpolate_2d(
                shallow_memory,
                (
                    int(current_memory.shape[-2]),
                    int(current_memory.shape[-1]),
                ),
            )

        previous_memory = _resize_image_memory(
            memory=previous_memory,
            reference=current_memory,
            fallback=shallow_memory,
            expected_channels=self.channels,
            module_name="MoR",
        )

        current_memory = (
            current_memory
            + self.position_mixer(
                current_memory
            )
        )

        shallow_memory = (
            shallow_memory
            + self.position_mixer(
                shallow_memory
            )
        )

        current_memory = _cast_tensor(
            current_memory,
            compute_dtype,
            compute_device,
        )

        shallow_memory = _cast_tensor(
            shallow_memory,
            compute_dtype,
            compute_device,
        )

        (
            text_tokens,
            text_mask,
            text_summary,
        ) = self._prepare_text(
            text_context=text_context,
            text_attention_mask=(
                text_attention_mask
            ),
            batch_size=batch_size,
            dtype=compute_dtype,
            device=compute_device,
        )

        shallow_tokens = shallow_memory.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            batch_size,
            -1,
            self.channels,
        )

        previous_tokens = previous_memory.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            batch_size,
            -1,
            self.channels,
        )

        context_parts = [
            shallow_tokens,
            previous_tokens,
        ]

        context_masks = [
            torch.ones(
                batch_size,
                shallow_tokens.shape[1],
                device=compute_device,
                dtype=torch.bool,
            ),
            torch.ones(
                batch_size,
                previous_tokens.shape[1],
                device=compute_device,
                dtype=torch.bool,
            ),
        ]

        if self.prototype_tokens.numel() > 0:
            prototypes = (
                self.prototype_tokens[
                    None
                ].expand(
                    batch_size,
                    -1,
                    -1,
                ).to(
                    device=compute_device,
                    dtype=compute_dtype,
                )
            )

            context_parts.append(
                prototypes
            )

            context_masks.append(
                torch.ones(
                    batch_size,
                    prototypes.shape[1],
                    device=compute_device,
                    dtype=torch.bool,
                )
            )

        if text_tokens is not None:
            context_parts.append(
                text_tokens
            )

            if text_mask is None:
                context_masks.append(
                    torch.ones(
                        batch_size,
                        text_tokens.shape[1],
                        device=compute_device,
                        dtype=torch.bool,
                    )
                )
            else:
                context_masks.append(
                    text_mask
                )

        context_tokens = torch.cat(
            context_parts,
            dim=1,
        )

        context_mask = torch.cat(
            context_masks,
            dim=1,
        )

        context_tokens = _cast_tensor(
            context_tokens,
            compute_dtype,
            compute_device,
        )

        recursive_state = current_memory
        recursion_states = []

        # 真正的权重共享：重复调用同一个 shared_cell。
        for _ in range(
            int(
                self.config.max_recursions
            )
        ):
            recursive_state = self.shared_cell(
                recursive_state,
                context_tokens,
                context_mask,
            )

            recursion_states.append(
                recursive_state
            )

        state_stack = torch.stack(
            recursion_states,
            dim=1,
        )

        state_stack = _cast_tensor(
            state_stack,
            compute_dtype,
            compute_device,
        )

        image_summary = current_memory.mean(
            dim=(-2, -1)
        )

        router_condition = (
            image_summary
            + timestep_condition
            + text_summary
        )

        router_condition = _cast_tensor(
            router_condition,
            compute_dtype,
            compute_device,
        )

        router_logits = self.router(
            router_condition
        )

        # Softmax 使用 FP32，之后恢复计算 dtype。
        router_weights = F.softmax(
            router_logits.float(),
            dim=-1,
        ).to(
            device=compute_device,
            dtype=compute_dtype,
        )

        mixed_memory = (
            state_stack
            * router_weights[
                :,
                :,
                None,
                None,
                None,
            ]
        ).sum(
            dim=1
        )

        mixed_memory = _cast_tensor(
            mixed_memory,
            compute_dtype,
            compute_device,
        )

        for unit in self.post_memory_units:
            mixed_memory = unit(
                mixed_memory
            )

        gate_values = self.gate_modulation(
            timestep_condition
            + text_summary
        )

        gate_values = _cast_tensor(
            gate_values,
            compute_dtype,
            compute_device,
        )

        (
            update_logits,
            output_gate,
        ) = gate_values.chunk(
            2,
            dim=-1,
        )

        update_strength = torch.sigmoid(
            update_logits
        )[:, :, None, None]

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
            + half
            * torch.tanh(
                output_gate
            )
        )[:, :, None, None]

        update_strength = _cast_tensor(
            update_strength,
            compute_dtype,
            compute_device,
        )

        output_strength = _cast_tensor(
            output_strength,
            compute_dtype,
            compute_device,
        )

        updated_memory = (
            previous_memory
            + update_strength
            * (
                mixed_memory
                - previous_memory
            )
        )

        updated_memory = _cast_tensor(
            updated_memory,
            compute_dtype,
            compute_device,
        )

        global_refinement = (
            updated_memory
            * output_strength
        )

        global_refinement = _safe_interpolate_2d(
            global_refinement,
            (
                int(height),
                int(width),
            ),
        )

        detail = self.detail_norm(
            current_full
        )

        detail = _cast_tensor(
            detail,
            compute_dtype,
            compute_device,
        )

        detail = (
            self.detail_depthwise_3(
                detail
            )
            + self.detail_depthwise_dilated(
                detail
            )
        )

        detail = F.gelu(
            detail,
            approximate="tanh",
        )

        detail = self.detail_proj(
            detail
        )

        detail = _cast_tensor(
            detail,
            compute_dtype,
            compute_device,
        )

        detail_scale = self.detail_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
        )

        fused = (
            global_refinement
            + detail_scale
            * detail
        )

        fused = _cast_tensor(
            fused,
            compute_dtype,
            compute_device,
        )

        residual = self.output_proj(
            fused
        )

        residual = _cast_tensor(
            residual,
            compute_dtype,
            compute_device,
        )

        residual = residual.permute(
            0,
            2,
            3,
            1,
        )[:, None].contiguous()

        branch_scale = self.branch_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        )

        output = (
            residual_input
            + branch_scale
            * residual
        )

        output = _cast_tensor(
            output,
            compute_dtype,
            compute_device,
        )

        _check_finite(
            output,
            "MoR 输出",
        )

        _check_finite(
            updated_memory,
            "MoR updated_memory",
        )

        return output, updated_memory


# ============================================================
# 默认 block placement
# ============================================================


def build_every_n_block_indices(
    num_blocks: int,
    stride: int = 4,
    include_last_partial: bool = False,
) -> List[int]:
    return _v2_runtime.build_every_n_block_indices(
        num_blocks=num_blocks,
        stride=stride,
        include_last_partial=(
            include_last_partial
        ),
    )


def build_group_block_indices(
    num_blocks: int,
    group_size: int,
    placement: str = "middle",
    include_partial_group: bool = False,
) -> List[int]:
    num_blocks = int(
        num_blocks
    )
    group_size = int(
        group_size
    )

    if group_size <= 0:
        raise ValueError(
            "group_size 必须大于 0"
        )

    if placement not in (
        "start",
        "middle",
        "end",
    ):
        raise ValueError(
            f"不支持的 placement：{placement}"
        )

    result: List[int] = []

    for start in range(
        0,
        num_blocks,
        group_size,
    ):
        group = list(
            range(
                start,
                min(
                    start + group_size,
                    num_blocks,
                ),
            )
        )

        if (
            len(group) < group_size
            and not include_partial_group
        ):
            continue

        if not group:
            continue

        if placement == "start":
            selected = group[0]
        elif placement == "end":
            selected = group[-1]
        else:
            selected = group[
                (len(group) - 1) // 2
            ]

        result.append(
            selected
        )

    return sorted(
        set(result)
    )


def build_stride_block_indices(
    num_blocks: int,
    stride: int,
    offset: int,
    include_last_block: bool = False,
) -> List[int]:
    num_blocks = int(
        num_blocks
    )
    stride = int(
        stride
    )

    if stride <= 0:
        raise ValueError(
            "stride 必须大于 0"
        )

    offset = int(
        offset
    ) % stride

    result = list(
        range(
            offset,
            num_blocks,
            stride,
        )
    )

    if not include_last_block:
        result = [
            index
            for index in result
            if index != num_blocks - 1
        ]

    return sorted(
        set(result)
    )


# ============================================================
# checkpoint / LoRA key 工具
# ============================================================


def _key_has_native_namespace(
    key: str,
    namespace: str,
) -> bool:
    key = str(
        key
    ).lower()

    return (
        f".{namespace}." in key
        or key.startswith(
            f"{namespace}."
        )
    )


def _key_has_lora_namespace(
    key: str,
    namespace: str,
) -> bool:
    key = str(
        key
    ).lower()

    return (
        f"_{namespace}_" in key
        or key.startswith(
            f"{namespace}_"
        )
        or f"{namespace}_" in key
    )


def _infer_indices_from_keys(
    source_keys: Sequence[str],
    namespace: str,
) -> List[int]:
    namespace_pattern = re.escape(
        namespace
    )

    patterns = (
        # PyTorch / Comfy 原生点路径。
        re.compile(
            rf"(?:^|\.)blocks\.(\d+)"
            rf"\.{namespace_pattern}(?:\.|$)",
            re.IGNORECASE,
        ),

        # LoRA 下划线路径。
        re.compile(
            rf"(?:^|_)blocks_(\d+)"
            rf"_{namespace_pattern}(?:_|$)",
            re.IGNORECASE,
        ),
    )

    indices = set()

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


def _validate_indices(
    indices: Sequence[int],
    num_blocks: int,
    family_name: str,
) -> List[int]:
    result = sorted(
        set(
            int(index)
            for index in indices
        )
    )

    invalid = [
        index
        for index in result
        if (
            index < 0
            or index >= num_blocks
        )
    ]

    if invalid:
        raise ValueError(
            f"{family_name} 包含无效 block index："
            f"{invalid}，模型总 block 数={num_blocks}"
        )

    return result


def _single_value(
    values,
    value_name: str,
):
    values = set(
        values
    )

    if len(values) > 1:
        raise RuntimeError(
            f"V4 checkpoint 包含多个不一致的 {value_name}："
            f"{sorted(values)}"
        )

    if not values:
        return None

    return next(
        iter(values)
    )


# ============================================================
# AttnRes config 推断
# ============================================================


def _infer_attnres_config_from_state_dict(
    source_state_dict: Optional[
        Dict[str, torch.Tensor]
    ],
    model_channels: int,
    default_config: AttnResRuntimeConfig,
) -> AttnResRuntimeConfig:
    config = copy.deepcopy(
        default_config
    )

    if not source_state_dict:
        return config

    channel_values = set()
    memory_unit_indices = set()
    kernel_values = set()
    prototype_values = set()

    saw_core_weight = False
    saw_prototype_parameter = False

    memory_unit_pattern = re.compile(
        r"\.spatial_graft\.memory_units\.(\d+)\.",
        re.IGNORECASE,
    )

    for raw_key, tensor in source_state_dict.items():
        if not torch.is_tensor(
            tensor
        ):
            continue

        key = str(
            raw_key
        ).lower()

        # 只从原生 base 权重推断，不从 LoRA A/B 低秩形状推断。
        if ".spatial_graft." not in key:
            continue

        if key.endswith(
            ".spatial_graft.current_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "AttnRes current_proj.weight 维度异常："
                    f"{raw_key} -> {tuple(tensor.shape)}"
                )

            if int(tensor.shape[1]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "AttnRes checkpoint model_channels 不一致："
                    f"{tensor.shape[1]} != {model_channels}"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            saw_core_weight = True

        elif key.endswith(
            ".spatial_graft.shallow_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "AttnRes shallow_proj.weight 维度异常"
                )

            if int(tensor.shape[1]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "AttnRes shallow_proj model_channels 不一致"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            saw_core_weight = True

        elif key.endswith(
            ".spatial_graft.output_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "AttnRes output_proj.weight 维度异常"
                )

            if int(tensor.shape[0]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "AttnRes output_proj model_channels 不一致"
                )

            channel_values.add(
                int(tensor.shape[1])
            )
            saw_core_weight = True

        elif key.endswith(
            ".spatial_graft.position_mixer.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "AttnRes position_mixer.weight 维度异常"
                )

            if int(tensor.shape[1]) != 1:
                raise RuntimeError(
                    "AttnRes position_mixer 不是 depthwise Conv2d"
                )

            if int(tensor.shape[2]) != int(
                tensor.shape[3]
            ):
                raise RuntimeError(
                    "AttnRes position_mixer kernel 不是正方形"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            kernel_values.add(
                int(tensor.shape[2])
            )
            saw_core_weight = True

        elif key.endswith(
            ".spatial_graft.prototype_tokens"
        ):
            if tensor.ndim != 2:
                raise RuntimeError(
                    "AttnRes prototype_tokens 维度异常"
                )

            prototype_values.add(
                int(tensor.shape[0])
            )
            channel_values.add(
                int(tensor.shape[1])
            )
            saw_prototype_parameter = True

        unit_match = memory_unit_pattern.search(
            key
        )

        if unit_match is not None:
            memory_unit_indices.add(
                int(
                    unit_match.group(1)
                )
            )

        if (
            ".spatial_graft.memory_units."
            in key
            and key.endswith(
                ".depthwise.weight"
            )
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "AttnRes memory depthwise 维度异常"
                )

            if int(tensor.shape[1]) != 1:
                raise RuntimeError(
                    "AttnRes memory depthwise 形状异常"
                )

            if int(tensor.shape[2]) != int(
                tensor.shape[3]
            ):
                raise RuntimeError(
                    "AttnRes memory kernel 不是正方形"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            kernel_values.add(
                int(tensor.shape[2])
            )

    channels = _single_value(
        channel_values,
        "AttnRes channels",
    )

    if channels is not None:
        config.attention_channels_override = int(
            channels
        )
        config.attention_ratio = (
            float(channels)
            / float(model_channels)
        )

    kernel_size = _single_value(
        kernel_values,
        "AttnRes spatial kernel",
    )

    if kernel_size is not None:
        config.spatial_kernel_size = int(
            kernel_size
        )

    prototype_count = _single_value(
        prototype_values,
        "AttnRes prototype count",
    )

    if prototype_count is not None:
        config.num_prototype_tokens = int(
            prototype_count
        )
    elif (
        saw_core_weight
        and saw_prototype_parameter
    ):
        config.num_prototype_tokens = 0

    if memory_unit_indices:
        config.memory_depth = (
            max(memory_unit_indices)
            + 1
        )
    elif saw_core_weight:
        # 完整 core checkpoint 中完全没有 memory_units 时可推断为 0。
        saw_any_memory_unit_key = any(
            ".spatial_graft.memory_units."
            in str(key).lower()
            for key in source_state_dict.keys()
        )

        if not saw_any_memory_unit_key:
            config.memory_depth = 0

    return config


# ============================================================
# MoR config 推断
# ============================================================


def _infer_mor_config_from_state_dict(
    source_state_dict: Optional[
        Dict[str, torch.Tensor]
    ],
    model_channels: int,
    context_dim: int,
    default_config: MoRRuntimeConfig,
) -> MoRRuntimeConfig:
    config = copy.deepcopy(
        default_config
    )

    if not source_state_dict:
        return config

    channel_values = set()
    kernel_values = set()
    prototype_values = set()
    recursion_values = set()
    memory_unit_indices = set()

    saw_core_weight = False
    saw_text_proj = False
    saw_prototype_parameter = False

    memory_unit_pattern = re.compile(
        r"\.mor_graft\.post_memory_units\.(\d+)\.",
        re.IGNORECASE,
    )

    for raw_key, tensor in source_state_dict.items():
        if not torch.is_tensor(
            tensor
        ):
            continue

        key = str(
            raw_key
        ).lower()

        if ".mor_graft." not in key:
            continue

        if key.endswith(
            ".mor_graft.current_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "MoR current_proj.weight 维度异常"
                )

            if int(tensor.shape[1]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "MoR checkpoint model_channels 不一致："
                    f"{tensor.shape[1]} != {model_channels}"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            saw_core_weight = True

        elif key.endswith(
            ".mor_graft.shallow_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "MoR shallow_proj.weight 维度异常"
                )

            if int(tensor.shape[1]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "MoR shallow_proj model_channels 不一致"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            saw_core_weight = True

        elif key.endswith(
            ".mor_graft.output_proj.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "MoR output_proj.weight 维度异常"
                )

            if int(tensor.shape[0]) != int(
                model_channels
            ):
                raise RuntimeError(
                    "MoR output_proj model_channels 不一致"
                )

            channel_values.add(
                int(tensor.shape[1])
            )
            saw_core_weight = True

        elif key.endswith(
            ".mor_graft.position_mixer.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "MoR position_mixer.weight 维度异常"
                )

            if int(tensor.shape[1]) != 1:
                raise RuntimeError(
                    "MoR position_mixer 不是 depthwise Conv2d"
                )

            if int(tensor.shape[2]) != int(
                tensor.shape[3]
            ):
                raise RuntimeError(
                    "MoR position_mixer kernel 不是正方形"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            kernel_values.add(
                int(tensor.shape[2])
            )
            saw_core_weight = True

        elif key.endswith(
            ".mor_graft.text_proj.weight"
        ):
            if tensor.ndim != 2:
                raise RuntimeError(
                    "MoR text_proj.weight 维度异常"
                )

            if int(tensor.shape[1]) != int(
                context_dim
            ):
                raise RuntimeError(
                    "MoR text context_dim 与当前模型不一致："
                    f"{tensor.shape[1]} != {context_dim}"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            saw_text_proj = True

        elif key.endswith(
            ".mor_graft.prototype_tokens"
        ):
            if tensor.ndim != 2:
                raise RuntimeError(
                    "MoR prototype_tokens 维度异常"
                )

            prototype_values.add(
                int(tensor.shape[0])
            )
            channel_values.add(
                int(tensor.shape[1])
            )
            saw_prototype_parameter = True

        elif key.endswith(
            ".mor_graft.router.3.weight"
        ):
            if tensor.ndim != 2:
                raise RuntimeError(
                    "MoR router.3.weight 维度异常"
                )

            recursion_values.add(
                int(tensor.shape[0])
            )
            channel_values.add(
                int(tensor.shape[1])
            )
            saw_core_weight = True

        elif key.endswith(
            ".mor_graft.router.3.bias"
        ):
            if tensor.ndim != 1:
                raise RuntimeError(
                    "MoR router.3.bias 维度异常"
                )

            recursion_values.add(
                int(tensor.shape[0])
            )

        elif key.endswith(
            ".mor_graft.shared_cell.local_depthwise.weight"
        ):
            if tensor.ndim != 4:
                raise RuntimeError(
                    "MoR local_depthwise.weight 维度异常"
                )

            if int(tensor.shape[1]) != 1:
                raise RuntimeError(
                    "MoR local_depthwise 不是 depthwise Conv2d"
                )

            if int(tensor.shape[2]) != int(
                tensor.shape[3]
            ):
                raise RuntimeError(
                    "MoR local kernel 不是正方形"
                )

            channel_values.add(
                int(tensor.shape[0])
            )
            kernel_values.add(
                int(tensor.shape[2])
            )

        unit_match = memory_unit_pattern.search(
            key
        )

        if unit_match is not None:
            memory_unit_indices.add(
                int(
                    unit_match.group(1)
                )
            )

    channels = _single_value(
        channel_values,
        "MoR channels",
    )

    if channels is not None:
        config.channels_override = int(
            channels
        )
        config.channel_ratio = (
            float(channels)
            / float(model_channels)
        )

    kernel_size = _single_value(
        kernel_values,
        "MoR spatial kernel",
    )

    if kernel_size is not None:
        config.spatial_kernel_size = int(
            kernel_size
        )

    recursion_count = _single_value(
        recursion_values,
        "MoR max_recursions",
    )

    if recursion_count is not None:
        if int(recursion_count) <= 0:
            raise RuntimeError(
                "MoR checkpoint max_recursions 非法"
            )

        config.max_recursions = int(
            recursion_count
        )

    prototype_count = _single_value(
        prototype_values,
        "MoR prototype count",
    )

    if prototype_count is not None:
        config.num_prototype_tokens = int(
            prototype_count
        )
    elif (
        saw_core_weight
        and saw_prototype_parameter
    ):
        config.num_prototype_tokens = 0

    if memory_unit_indices:
        config.memory_depth = (
            max(memory_unit_indices)
            + 1
        )
    elif saw_core_weight:
        saw_any_memory_unit_key = any(
            ".mor_graft.post_memory_units."
            in str(key).lower()
            for key in source_state_dict.keys()
        )

        if not saw_any_memory_unit_key:
            config.memory_depth = 0

    # 只有在检测到完整 MoR core 权重时，才根据 text_proj 是否存在
    # 推断关闭 text conditioning。LoRA 可能只包含少量目标层，因此
    # 不能根据 LoRA 中缺少 text_proj 判断为 False。
    if saw_text_proj:
        config.use_text_conditioning = True
    elif saw_core_weight:
        direct_core_count = sum(
            1
            for key in source_state_dict.keys()
            if (
                ".mor_graft.current_proj.weight"
                in str(key).lower()
                or ".mor_graft.output_proj.weight"
                in str(key).lower()
                or ".mor_graft.router.3.weight"
                in str(key).lower()
            )
        )

        if direct_core_count >= 3:
            config.use_text_conditioning = False

    return config


# ============================================================
# Runtime context
# ============================================================


def _new_runtime_context(
    text_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    return {
        "shallow": None,
        "mudd_memory": None,
        "attnres_memory": None,
        "mor_memory": None,
        "text_context": None,
        "text_mask": text_mask,
    }


def _get_or_create_runtime_context(
    model: nn.Module,
) -> Dict[str, Any]:
    context = getattr(
        model,
        "_anima_grafted4_runtime_context",
        None,
    )

    if context is None:
        context = _new_runtime_context()

        object.__setattr__(
            model,
            "_anima_grafted4_runtime_context",
            context,
        )

    return context


def _extract_primary_tensor(
    output,
) -> Tuple[torch.Tensor, Optional[str]]:
    if torch.is_tensor(
        output
    ):
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
        "[SpatialGraftV4] 原始 Anima block.forward 返回了"
        f"不支持的类型：{type(output).__name__}。"
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


def _extract_block_text_context(
    args,
    kwargs,
) -> Optional[torch.Tensor]:
    """
    原版 Anima Block.forward：

        block(
            hidden_states,
            timestep_embedding,
            crossattn_emb,
            attn_params,
            ...
        )

    包装器已经显式接收前两个参数，因此 args[0] 通常就是
    crossattn_emb。
    """

    for name in (
        "crossattn_emb",
        "context",
        "encoder_hidden_states",
    ):
        value = kwargs.get(
            name
        )

        if torch.is_tensor(
            value
        ):
            return value

    if (
        len(args) > 0
        and torch.is_tensor(args[0])
    ):
        return args[0]

    return None


def _branch_delta(
    base_output: torch.Tensor,
    branch_input: torch.Tensor,
    branch_output: torch.Tensor,
    branch_name: str,
) -> torch.Tensor:
    """
    branch 在其权重 dtype 中计算。

    差值先在 branch dtype 中形成，再恢复到原始 block 输出的
    dtype/device。这样：

      - 不会因为 graft 权重是 FP16/BF16 而永久改变主干精度；
      - 不会因为 LayerScale 为 FP32 而把主干 residual 提升为 FP32；
      - output_proj 为零时 delta 精确为零。
    """

    if branch_output.shape != branch_input.shape:
        raise ValueError(
            f"{branch_name} 输出形状与输入不一致："
            f"{tuple(branch_output.shape)} != "
            f"{tuple(branch_input.shape)}"
        )

    delta = (
        branch_output
        - branch_input
    )

    delta = delta.to(
        device=base_output.device,
        dtype=base_output.dtype,
    )

    _check_finite(
        delta,
        f"{branch_name} delta",
    )

    return delta


# ============================================================
# Block forward 安装
# ============================================================


def _infer_extra_block_count(
    source_keys: Sequence[str],
) -> int:
    indices = set()
    dotted_pattern = re.compile(
        r"(?:^|\.)extra_blocks\.(\d+)\."
    )
    adapter_pattern = re.compile(
        r"(?:^|_)extra_blocks_(\d+)_"
    )

    for raw_key in source_keys or []:
        key = str(raw_key).replace("/", ".").lower()
        match = dotted_pattern.search(key)
        if match is None:
            match = adapter_pattern.search(key)
        if match is not None:
            indices.add(int(match.group(1)))

    if not indices:
        return 0

    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise RuntimeError(
            "extra_blocks checkpoint 索引不连续："
            f"found={sorted(indices)}, expected={sorted(expected)}"
        )
    return len(indices)


def _resolve_extra_block_layout(
    num_base_blocks: int,
    config: ExtraBlockRuntimeConfig,
) -> Tuple[List[int], List[int]]:
    count = int(config.num_blocks)
    if count < 0:
        raise ValueError("extra block 数量不能为负数")
    if count == 0 or not config.enabled:
        return [], []

    if config.insert_after:
        insert_after = [int(index) for index in config.insert_after]
        if len(insert_after) != count:
            raise ValueError("extra block insert_after 数量与 num_blocks 不一致")
    else:
        insert_after = [
            max(
                0,
                min(
                    num_base_blocks - 1,
                    ((index + 1) * num_base_blocks) // (count + 1) - 1,
                ),
            )
            for index in range(count)
        ]

    if config.source_indices:
        source_indices = [int(index) for index in config.source_indices]
        if len(source_indices) != count:
            raise ValueError("extra block source_indices 数量与 num_blocks 不一致")
    else:
        source_indices = list(insert_after)

    insert_after = _validate_indices(
        insert_after,
        num_base_blocks,
        "Extra block insertion indices",
    )
    source_indices = _validate_indices(
        source_indices,
        num_base_blocks,
        "Extra block source indices",
    )
    return insert_after, source_indices


def _install_extra_blocks(
    model: nn.Module,
    config: ExtraBlockRuntimeConfig,
):
    if hasattr(model, "extra_blocks"):
        raise RuntimeError("Extra Anima blocks 已经安装")

    insert_after, source_indices = _resolve_extra_block_layout(
        len(model.blocks),
        config,
    )
    extra_blocks = []
    global_reference = _find_model_reference_parameter(model)

    for insertion_index, source_index in zip(
        insert_after,
        source_indices,
    ):
        cloned_block = copy.deepcopy(model.blocks[source_index])
        for graft_name in ("mudd_graft", "spatial_graft", "mor_graft"):
            if graft_name in cloned_block._modules:
                del cloned_block._modules[graft_name]

        extra_block = ExtraAnimaBlock(
            block=cloned_block,
            insert_after=insertion_index,
            source_index=source_index,
            residual_scale_init=config.residual_scale_init,
        )
        reference = _find_block_reference_parameter(
            model.blocks[source_index],
            global_reference,
        )
        _move_module_like_parameter(extra_block, reference)
        extra_blocks.append(extra_block)

    model.extra_blocks = nn.ModuleList(extra_blocks)
    object.__setattr__(model, "extra_block_config", config)
    object.__setattr__(model, "extra_block_insert_after", insert_after)
    object.__setattr__(model, "extra_block_source_indices", source_indices)
    object.__setattr__(
        model,
        "_anima_extra_blocks_by_position",
        {
            position: [
                block
                for block in model.extra_blocks
                if int(block.insert_after) == position
            ]
            for position in sorted(set(insert_after))
        },
    )


def _install_grafted4_block_forward(
    block: nn.Module,
    model: nn.Module,
    block_index: int,
    apply_mudd: bool,
    apply_attnres: bool,
    apply_mor: bool,
):
    if getattr(
        block,
        "_anima_grafted4_block_forward_installed",
        False,
    ):
        object.__setattr__(
            block,
            "_anima_grafted4_apply_mudd",
            bool(
                getattr(
                    block,
                    "_anima_grafted4_apply_mudd",
                    False,
                )
                or apply_mudd
            ),
        )

        object.__setattr__(
            block,
            "_anima_grafted4_apply_attnres",
            bool(
                getattr(
                    block,
                    "_anima_grafted4_apply_attnres",
                    False,
                )
                or apply_attnres
            ),
        )

        object.__setattr__(
            block,
            "_anima_grafted4_apply_mor",
            bool(
                getattr(
                    block,
                    "_anima_grafted4_apply_mor",
                    False,
                )
                or apply_mor
            ),
        )

        return

    if (
        apply_mudd
        and not hasattr(
            block,
            "mudd_graft",
        )
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 mudd_graft"
        )

    if (
        apply_attnres
        and not hasattr(
            block,
            "spatial_graft",
        )
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 spatial_graft"
        )

    if (
        apply_mor
        and not hasattr(
            block,
            "mor_graft",
        )
    ):
        raise RuntimeError(
            f"Block {block_index} 尚未添加 mor_graft"
        )

    original_forward = block.forward

    object.__setattr__(
        block,
        "_anima_grafted4_original_forward",
        original_forward,
    )

    object.__setattr__(
        block,
        "_anima_grafted4_owner_model_ref",
        weakref.ref(model),
    )

    object.__setattr__(
        block,
        "_anima_grafted4_block_index",
        int(block_index),
    )

    object.__setattr__(
        block,
        "_anima_grafted4_apply_mudd",
        bool(apply_mudd),
    )

    object.__setattr__(
        block,
        "_anima_grafted4_apply_attnres",
        bool(apply_attnres),
    )

    object.__setattr__(
        block,
        "_anima_grafted4_apply_mor",
        bool(apply_mor),
    )

    def grafted4_block_forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        *args,
        **kwargs,
    ):
        owner_ref = getattr(
            self,
            "_anima_grafted4_owner_model_ref",
            None,
        )

        owner_model = (
            owner_ref()
            if owner_ref is not None
            else None
        )

        if owner_model is None:
            raise RuntimeError(
                "[SpatialGraftV4] 无法取得所属 Anima model"
            )

        context = _get_or_create_runtime_context(
            owner_model
        )

        runtime_block_index = int(
            getattr(
                self,
                "_anima_grafted4_block_index",
                -1,
            )
        )

        # 每次完整前向进入 block 0 时重置所有 persistent memory。
        if runtime_block_index == 0:
            if not torch.is_tensor(
                x_B_T_H_W_D
            ):
                raise TypeError(
                    "Anima block 0 输入不是 Tensor"
                )

            context["shallow"] = (
                x_B_T_H_W_D
            )
            context["mudd_memory"] = None
            context["attnres_memory"] = None
            context["mor_memory"] = None
            context["text_context"] = None

        original_output = (
            self._anima_grafted4_original_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                *args,
                **kwargs,
            )
        )

        base_output, container_type = (
            _extract_primary_tensor(
                original_output
            )
        )

        if not torch.is_tensor(
            base_output
        ):
            raise TypeError(
                "Anima block 主输出不是 Tensor"
            )

        shallow = context.get(
            "shallow"
        )

        if shallow is None:
            raise RuntimeError(
                "[SpatialGraftV4] 未捕获 shallow feature。"
                "Anima blocks 可能未从 block 0 顺序执行。"
            )

        text_context = _extract_block_text_context(
            args,
            kwargs,
        )

        if text_context is not None:
            context["text_context"] = (
                text_context
            )

        combined_output = base_output

        apply_mudd_runtime = bool(
            getattr(
                self,
                "_anima_grafted4_apply_mudd",
                False,
            )
        )

        apply_attnres_runtime = bool(
            getattr(
                self,
                "_anima_grafted4_apply_attnres",
                False,
            )
        )

        apply_mor_runtime = bool(
            getattr(
                self,
                "_anima_grafted4_apply_mor",
                False,
            )
        )

        # ----------------------------------------------------
        # MUDD：读取 base_output，不读取其他 graft 输出
        # ----------------------------------------------------

        if apply_mudd_runtime:
            mudd_dtype = (
                self.mudd_graft
                .current_memory_proj
                .weight
                .dtype
            )

            mudd_device = (
                self.mudd_graft
                .current_memory_proj
                .weight
                .device
            )

            _require_same_device(
                base_output,
                mudd_device,
                "MUDD 所在 block 输出",
            )

            mudd_input = base_output.to(
                device=mudd_device,
                dtype=mudd_dtype,
            )

            mudd_shallow = shallow.to(
                device=mudd_device,
                dtype=mudd_dtype,
            )

            mudd_timestep = emb_B_T_D.to(
                device=mudd_device,
                dtype=mudd_dtype,
            )

            (
                mudd_output,
                mudd_memory,
            ) = self.mudd_graft(
                mudd_input,
                mudd_shallow,
                context.get(
                    "mudd_memory"
                ),
                mudd_timestep,
            )

            context["mudd_memory"] = (
                mudd_memory
            )

            combined_output = (
                combined_output
                + _branch_delta(
                    base_output,
                    mudd_input,
                    mudd_output,
                    "MUDD",
                )
            )

        # ----------------------------------------------------
        # AttnRes：读取同一个 base_output
        # ----------------------------------------------------

        if apply_attnres_runtime:
            attnres_dtype = (
                self.spatial_graft
                .current_proj
                .weight
                .dtype
            )

            attnres_device = (
                self.spatial_graft
                .current_proj
                .weight
                .device
            )

            _require_same_device(
                base_output,
                attnres_device,
                "AttnRes 所在 block 输出",
            )

            attnres_input = base_output.to(
                device=attnres_device,
                dtype=attnres_dtype,
            )

            attnres_shallow = shallow.to(
                device=attnres_device,
                dtype=attnres_dtype,
            )

            attnres_timestep = emb_B_T_D.to(
                device=attnres_device,
                dtype=attnres_dtype,
            )

            (
                attnres_output,
                attnres_memory,
            ) = self.spatial_graft(
                attnres_input,
                attnres_shallow,
                context.get(
                    "attnres_memory"
                ),
                attnres_timestep,
            )

            context["attnres_memory"] = (
                attnres_memory
            )

            combined_output = (
                combined_output
                + _branch_delta(
                    base_output,
                    attnres_input,
                    attnres_output,
                    "AttnRes",
                )
            )

        # ----------------------------------------------------
        # MoR：读取同一个 base_output
        # ----------------------------------------------------

        if apply_mor_runtime:
            mor_dtype = (
                self.mor_graft
                .current_proj
                .weight
                .dtype
            )

            mor_device = (
                self.mor_graft
                .current_proj
                .weight
                .device
            )

            _require_same_device(
                base_output,
                mor_device,
                "MoR 所在 block 输出",
            )

            mor_input = base_output.to(
                device=mor_device,
                dtype=mor_dtype,
            )

            mor_shallow = shallow.to(
                device=mor_device,
                dtype=mor_dtype,
            )

            mor_timestep = emb_B_T_D.to(
                device=mor_device,
                dtype=mor_dtype,
            )

            mor_text_context = context.get(
                "text_context"
            )

            if mor_text_context is not None:
                mor_text_context = mor_text_context.to(
                    device=mor_device,
                    dtype=mor_dtype,
                )

            mor_text_mask = context.get(
                "text_mask"
            )

            if (
                mor_text_mask is not None
                and torch.is_tensor(
                    mor_text_mask
                )
            ):
                mor_text_mask = mor_text_mask.to(
                    device=mor_device
                )

            (
                mor_output,
                mor_memory,
            ) = self.mor_graft(
                mor_input,
                mor_shallow,
                context.get(
                    "mor_memory"
                ),
                mor_timestep,
                mor_text_context,
                mor_text_mask,
            )

            context["mor_memory"] = (
                mor_memory
            )

            combined_output = (
                combined_output
                + _branch_delta(
                    base_output,
                    mor_input,
                    mor_output,
                    "MoR",
                )
            )

        combined_output = combined_output.to(
            device=base_output.device,
            dtype=base_output.dtype,
        )

        extra_blocks = getattr(
            owner_model,
            "_anima_extra_blocks_by_position",
            {},
        ).get(runtime_block_index, ())
        for extra_block in extra_blocks:
            extra_reference = next(extra_block.block.parameters())
            extra_dtype = extra_reference.dtype
            extra_device = extra_reference.device
            _require_same_device(
                combined_output,
                extra_device,
                "Extra Anima block 输入",
            )
            extra_input = combined_output.to(
                device=extra_device,
                dtype=extra_dtype,
            )
            combined_output = extra_block(
                extra_input,
                emb_B_T_D.to(
                    device=extra_device,
                    dtype=extra_dtype,
                ),
                *args,
                **kwargs,
            ).to(
                device=base_output.device,
                dtype=base_output.dtype,
            )

        _check_finite(
            combined_output,
            "V4 combined block output",
        )

        return _replace_primary_tensor(
            original_output,
            combined_output,
            container_type,
        )

    block.forward = MethodType(
        grafted4_block_forward,
        block,
    )

    object.__setattr__(
        block,
        "_anima_grafted4_block_forward_installed",
        True,
    )


# ============================================================
# 顶层 runtime context forward
# ============================================================


def _get_positional_argument(
    args,
    index: int,
):
    if len(args) > index:
        return args[index]

    return None


def _install_model_runtime_context_forward(
    model: nn.Module,
):
    """
    包装 forward_mini_train_dit，只管理本次推理的 runtime context。

    不复制原版 Anima 顶层 forward，因此能够继续兼容：

      - attention mode；
      - split attention；
      - block swap；
      - LLM adapter；
      - ComfyUI 后续版本增加的参数。
    """

    if getattr(
        model,
        "_anima_grafted4_model_forward_installed",
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
            "Anima model 不存在 forward_mini_train_dit 或 forward"
        )

    original_forward = getattr(
        model,
        forward_name,
    )

    object.__setattr__(
        model,
        "_anima_grafted4_original_model_forward",
        original_forward,
    )

    object.__setattr__(
        model,
        "_anima_grafted4_wrapped_forward_name",
        forward_name,
    )

    sentinel = object()

    def grafted4_model_forward(
        self,
        *args,
        **kwargs,
    ):
        previous_context = getattr(
            self,
            "_anima_grafted4_runtime_context",
            sentinel,
        )

        # forward_mini_train_dit 的标准参数位置：
        #
        # 0 x
        # 1 timesteps
        # 2 crossattn_emb
        # 3 fps
        # 4 padding_mask
        # 5 source_attention_mask
        # 6 t5_input_ids
        # 7 t5_attn_mask

        source_attention_mask = kwargs.get(
            "source_attention_mask",
            _get_positional_argument(
                args,
                5,
            ),
        )

        t5_input_ids = kwargs.get(
            "t5_input_ids",
            _get_positional_argument(
                args,
                6,
            ),
        )

        t5_attn_mask = kwargs.get(
            "t5_attn_mask",
            _get_positional_argument(
                args,
                7,
            ),
        )

        if t5_input_ids is None:
            # 部分 ComfyUI 调用使用 target_* 名称。
            t5_input_ids = kwargs.get(
                "target_input_ids"
            )

        if t5_attn_mask is None:
            t5_attn_mask = kwargs.get(
                "target_attention_mask"
            )

        use_llm_adapter = bool(
            getattr(
                self,
                "use_llm_adapter",
                False,
            )
        )

        if (
            t5_input_ids is not None
            and use_llm_adapter
            and t5_attn_mask is not None
        ):
            text_mask = t5_attn_mask
        else:
            text_mask = source_attention_mask

        object.__setattr__(
            self,
            "_anima_grafted4_runtime_context",
            _new_runtime_context(
                text_mask=text_mask
            ),
        )

        try:
            return (
                self._anima_grafted4_original_model_forward(
                    *args,
                    **kwargs,
                )
            )
        finally:
            if previous_context is sentinel:
                try:
                    object.__delattr__(
                        self,
                        "_anima_grafted4_runtime_context",
                    )
                except AttributeError:
                    pass
            else:
                object.__setattr__(
                    self,
                    "_anima_grafted4_runtime_context",
                    previous_context,
                )

    setattr(
        model,
        forward_name,
        MethodType(
            grafted4_model_forward,
            model,
        ),
    )

    object.__setattr__(
        model,
        "_anima_grafted4_model_forward_installed",
        True,
    )


# ============================================================
# Mutation 主类
# ============================================================


class GraftedAnima:
    """
    AnimaBaker Mutation 固定入口。
    """

    MUTATION_API_VERSION = 1

    MUTATION_ID = (
        "anima_spatial_graft_v4"
    )

    DISPLAY_NAME = (
        "Anima Hybrid Image Graft V4 "
        "(MUDD + AttnRes + MoR)"
    )

    MUDD_NAMESPACE = "mudd_graft"
    ATTNRES_NAMESPACE = "spatial_graft"
    MOR_NAMESPACE = "mor_graft"

    MODULE_NAMESPACE = MOR_NAMESPACE

    DEFAULT_MUDD_CONFIG = MUDDGraftRuntimeConfig(
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

    DEFAULT_ATTNRES_CONFIG = AttnResRuntimeConfig(
        enabled=True,
        group_size=3,
        include_partial_group=False,
        placement="middle",
        attention_ratio=0.0625,
        attention_channels_override=None,
        memory_pool=4,
        max_memory_tokens=128,
        num_heads=4,
        memory_depth=1,
        spatial_kernel_size=3,
        dropout=0.0,
        attention_layer_scale_init=0.1,
        memory_layer_scale_init=0.1,
        branch_scale_init=1.0,
        output_init_std=0.0,
        memory_update_bias=-1.5,
        num_prototype_tokens=16,
        use_rms_norm=True,
        strict_image_only=True,
    )

    DEFAULT_MOR_CONFIG = MoRRuntimeConfig(
        enabled=True,
        block_stride=4,
        block_offset=2,
        include_last_block=False,
        channel_ratio=0.0625,
        channels_override=None,
        num_heads=4,
        max_recursions=3,
        max_memory_tokens=128,
        memory_pool=4,
        memory_depth=1,
        spatial_kernel_size=3,
        dilation=2,
        dropout=0.0,
        recurrent_layer_scale_init=0.1,
        local_layer_scale_init=0.1,
        branch_scale_init=1.0,
        output_init_std=0.0,
        memory_update_bias=-1.5,
        num_prototype_tokens=24,
        use_text_conditioning=True,
        max_text_tokens=128,
        use_rms_norm=True,
        strict_image_only=True,
    )

    @classmethod
    def detect(
        cls,
        state_dict_keys,
    ) -> int:
        """
        V4 检测策略。

        分数设计：

          - 专用 MUTATION_ID：240
          - MUDD + AttnRes + MoR 完整组合：220
          - AttnRes + MoR：205
          - MUDD + MoR：200
          - 任何 MoR 命名空间：190
          - MUDD + AttnRes：160
          - 仅 AttnRes：125
          - 仅 MUDD：80

        因此在 V4 checkpoint/LoRA 中，只要出现 V4 独有的
        mor_graft，V4 分数就显著高于 V2/V2G。

        仅 MUDD checkpoint 时返回 80，低于 V2 的明确匹配分数，
        避免将纯 V2 MUDD checkpoint 错判为 V4。
        """

        saw_mudd = False
        saw_attnres = False
        saw_mor = False
        saw_extra_blocks = False
        saw_v4_id = False

        saw_attnres_prototype = False
        saw_mor_router = False
        saw_mor_shared_cell = False

        for raw_key in state_dict_keys or []:
            key = str(
                raw_key
            ).lower()

            if cls.MUTATION_ID in key:
                saw_v4_id = True

            if (
                ".extra_blocks." in key
                or key.startswith("extra_blocks.")
                or "_extra_blocks_" in key
                or key.startswith("extra_blocks_")
            ):
                saw_extra_blocks = True

            if (
                ".mudd_graft." in key
                or key.startswith(
                    "mudd_graft."
                )
                or "_mudd_graft_" in key
                or "mudd_graft_" in key
            ):
                saw_mudd = True

            if (
                ".spatial_graft." in key
                or key.startswith(
                    "spatial_graft."
                )
                or "_spatial_graft_" in key
                or "spatial_graft_" in key
            ):
                saw_attnres = True

            if (
                ".mor_graft." in key
                or key.startswith(
                    "mor_graft."
                )
                or "_mor_graft_" in key
                or "mor_graft_" in key
            ):
                saw_mor = True

            if (
                "spatial_graft.prototype_tokens"
                in key
                or "spatial_graft_prototype_tokens"
                in key
            ):
                saw_attnres_prototype = True

            if (
                "mor_graft.router"
                in key
                or "mor_graft_router"
                in key
            ):
                saw_mor_router = True

            if (
                "mor_graft.shared_cell"
                in key
                or "mor_graft_shared_cell"
                in key
            ):
                saw_mor_shared_cell = True

        if saw_v4_id:
            return 240

        if saw_extra_blocks:
            return 235

        if (
            saw_mudd
            and saw_attnres
            and saw_mor
        ):
            return 220

        if (
            saw_attnres
            and saw_mor
        ):
            return 205

        if (
            saw_mudd
            and saw_mor
        ):
            return 200

        if saw_mor:
            # MoR 是 V4 的专有 family。router/shared_cell 组合进一步
            # 证明它不是其他同名的普通模块。
            if (
                saw_mor_router
                and saw_mor_shared_cell
            ):
                return 195

            return 190

        if (
            saw_mudd
            and saw_attnres
        ):
            return 160

        if saw_attnres:
            if saw_attnres_prototype:
                return 130

            return 125

        if saw_mudd:
            return 80

        return 0

    @classmethod
    def is_mutation_key(
        cls,
        key,
    ) -> bool:
        key = str(
            key
        ).lower()

        return (
            cls.MUTATION_ID in key

            or ".mudd_graft." in key
            or key.startswith(
                "mudd_graft."
            )
            or "_mudd_graft_" in key
            or "mudd_graft_" in key

            or ".spatial_graft." in key
            or key.startswith(
                "spatial_graft."
            )
            or "_spatial_graft_" in key
            or "spatial_graft_" in key

            or ".mor_graft." in key
            or key.startswith(
                "mor_graft."
            )
            or "_mor_graft_" in key
            or "mor_graft_" in key

            or ".extra_blocks." in key
            or key.startswith("extra_blocks.")
            or "_extra_blocks_" in key
            or key.startswith("extra_blocks_")
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
        直接复用 V2 的 MUDD checkpoint 形状推断，保证 MUDD
        runtime 参数结构和 V2 完全一致。
        """

        infer_method = getattr(
            _V2Mutation,
            "_infer_config_from_state_dict",
            None,
        )

        if infer_method is None:
            return copy.deepcopy(
                cls.DEFAULT_MUDD_CONFIG
            )

        return infer_method(
            source_state_dict,
            model_channels=int(
                model_channels
            ),
        )

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
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        原地安装 V4 hybrid image graft。

        source_keys:
            来自基础模型或 LoRA 的参数键，用于恢复三个 graft family
            的 block placement。

        source_state_dict:
            如果 AnimaBaker 能提供完整参数字典，则进一步恢复：
              - MUDD memory/detail channels；
              - MUDD memory depth/kernel；
              - AttnRes channels/depth/kernel/prototype count；
              - MoR channels/depth/kernel/prototype count；
              - MoR recursion count；
              - MoR text-conditioning 开关。
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

        first_block = model.blocks[0]

        cross_attn = getattr(
            first_block,
            "cross_attn",
            None,
        )

        if cross_attn is None:
            raise AttributeError(
                "Anima block 不存在 cross_attn，无法恢复 MoR context_dim"
            )

        context_dim = getattr(
            cross_attn,
            "context_dim",
            None,
        )

        if context_dim is None:
            context_dim = getattr(
                cross_attn,
                "_context_dim",
                None,
            )

        if context_dim is None:
            k_proj = getattr(
                cross_attn,
                "k_proj",
                None,
            )

            if (
                k_proj is not None
                and hasattr(
                    k_proj,
                    "in_features",
                )
            ):
                context_dim = int(
                    k_proj.in_features
                )

        if context_dim is None:
            raise AttributeError(
                "无法从 Anima cross_attn 推断 context_dim"
            )

        context_dim = int(
            context_dim
        )

        all_source_keys = list(
            source_keys or []
        )

        if source_state_dict:
            all_source_keys.extend(
                source_state_dict.keys()
            )

        extra_block_count = _infer_extra_block_count(
            all_source_keys
        )
        extra_payload = (
            runtime_config.get("extra_block_config")
            if isinstance(runtime_config, dict)
            else None
        )
        if extra_payload is not None:
            if not isinstance(extra_payload, dict):
                raise TypeError("extra_block_config 必须是 JSON object")
            extra_payload = dict(extra_payload)
            extra_payload["insert_after"] = tuple(
                int(index)
                for index in extra_payload.get("insert_after", ())
            )
            extra_payload["source_indices"] = tuple(
                int(index)
                for index in extra_payload.get("source_indices", ())
            )
            extra_block_config = ExtraBlockRuntimeConfig(**extra_payload)
            if int(extra_block_config.num_blocks) != extra_block_count:
                raise RuntimeError(
                    "Grafted4 配置与 checkpoint 的 extra block 数量不一致："
                    f"config={extra_block_config.num_blocks}, "
                    f"checkpoint={extra_block_count}"
                )
            if bool(extra_block_config.enabled) != (extra_block_count > 0):
                raise RuntimeError(
                    "Grafted4 配置中的 extra_block_config.enabled "
                    "与 checkpoint 拓扑不一致"
                )
        else:
            extra_block_config = ExtraBlockRuntimeConfig(
                enabled=extra_block_count > 0,
                num_blocks=extra_block_count,
            )
        _install_extra_blocks(
            model,
            extra_block_config,
        )

        # A sidecar configuration is authoritative for placement.  Checkpoint
        # key inference is only a fallback for legacy files without JSON.
        configured_indices = (
            runtime_config
            if isinstance(runtime_config, dict)
            else {}
        )
        configured_mudd_indices = configured_indices.get(
            "mudd_block_indices"
        )
        configured_attnres_indices = configured_indices.get(
            "attnres_block_indices"
        )
        configured_mor_indices = configured_indices.get(
            "mor_block_indices"
        )

        # ----------------------------------------------------
        # 恢复三个 family 的 block placement
        # ----------------------------------------------------

        mudd_indices = (
            [int(index) for index in configured_mudd_indices]
            if configured_mudd_indices is not None
            else _infer_indices_from_keys(
                all_source_keys,
                cls.MUDD_NAMESPACE,
            )
        )

        attnres_indices = (
            [int(index) for index in configured_attnres_indices]
            if configured_attnres_indices is not None
            else _infer_indices_from_keys(
                all_source_keys,
                cls.ATTNRES_NAMESPACE,
            )
        )

        mor_indices = (
            [int(index) for index in configured_mor_indices]
            if configured_mor_indices is not None
            else _infer_indices_from_keys(
                all_source_keys,
                cls.MOR_NAMESPACE,
            )
        )

        if configured_mudd_indices is None and not mudd_indices:
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

        if configured_attnres_indices is None and not attnres_indices:
            attnres_indices = (
                build_group_block_indices(
                    num_blocks=num_blocks,
                    group_size=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .group_size
                    ),
                    placement=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .placement
                    ),
                    include_partial_group=(
                        cls.DEFAULT_ATTNRES_CONFIG
                        .include_partial_group
                    ),
                )
            )

        if configured_mor_indices is None and not mor_indices:
            mor_indices = (
                build_stride_block_indices(
                    num_blocks=num_blocks,
                    stride=(
                        cls.DEFAULT_MOR_CONFIG
                        .block_stride
                    ),
                    offset=(
                        cls.DEFAULT_MOR_CONFIG
                        .block_offset
                    ),
                    include_last_block=(
                        cls.DEFAULT_MOR_CONFIG
                        .include_last_block
                    ),
                )
            )

        mudd_indices = _validate_indices(
            mudd_indices,
            num_blocks,
            "MUDD indices",
        )

        attnres_indices = _validate_indices(
            attnres_indices,
            num_blocks,
            "AttnRes indices",
        )

        mor_indices = _validate_indices(
            mor_indices,
            num_blocks,
            "MoR indices",
        )

        if not (
            mudd_indices
            or attnres_indices
            or mor_indices
            or extra_block_count
        ):
            raise RuntimeError(
                "V4 没有任何可安装的 graft block"
            )

        # ----------------------------------------------------
        # 恢复运行配置
        # ----------------------------------------------------

        mudd_config = cls._infer_mudd_config(
            source_state_dict,
            model_channels=model_channels,
        )

        # V4 image-only MUDD。
        if hasattr(
            mudd_config,
            "temporal_kernel_size",
        ):
            # 如果 checkpoint 明确推断出其他 temporal kernel，
            # 保留 checkpoint 结构以确保 shape 可加载；默认则为 1。
            mudd_config.temporal_kernel_size = int(
                mudd_config.temporal_kernel_size
            )

        attnres_config = (
            _infer_attnres_config_from_state_dict(
                source_state_dict=(
                    source_state_dict
                ),
                model_channels=model_channels,
                default_config=(
                    cls.DEFAULT_ATTNRES_CONFIG
                ),
            )
        )

        mor_config = (
            _infer_mor_config_from_state_dict(
                source_state_dict=(
                    source_state_dict
                ),
                model_channels=model_channels,
                context_dim=context_dim,
                default_config=(
                    cls.DEFAULT_MOR_CONFIG
                ),
            )
        )

        global_reference = (
            _find_model_reference_parameter(
                model
            )
        )

        mudd_set = set(
            mudd_indices
        )
        attnres_set = set(
            attnres_indices
        )
        mor_set = set(
            mor_indices
        )

        # ----------------------------------------------------
        # 添加 MUDD modules
        # ----------------------------------------------------

        for block_index in sorted(
            mudd_set
        ):
            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.MUDD_NAMESPACE,
            ):
                existing = getattr(
                    block,
                    cls.MUDD_NAMESPACE,
                )

                if not isinstance(
                    existing,
                    MUDDFormerGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.MUDD_NAMESPACE}："
                        f"{type(existing).__name__}"
                    )
            else:
                graft = MUDDFormerGraft(
                    model_channels=model_channels,
                    config=mudd_config,
                    block_index=block_index,
                )

                reference = (
                    _find_block_reference_parameter(
                        block,
                        global_reference,
                    )
                )

                _move_module_like_parameter(
                    graft,
                    reference,
                )

                block.add_module(
                    cls.MUDD_NAMESPACE,
                    graft,
                )

        # ----------------------------------------------------
        # 添加 AttnRes modules
        # ----------------------------------------------------

        for block_index in sorted(
            attnres_set
        ):
            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.ATTNRES_NAMESPACE,
            ):
                existing = getattr(
                    block,
                    cls.ATTNRES_NAMESPACE,
                )

                if not isinstance(
                    existing,
                    ImageAttnResGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.ATTNRES_NAMESPACE}："
                        f"{type(existing).__name__}"
                    )
            else:
                graft = ImageAttnResGraft(
                    model_channels=model_channels,
                    config=attnres_config,
                    block_index=block_index,
                )

                reference = (
                    _find_block_reference_parameter(
                        block,
                        global_reference,
                    )
                )

                _move_module_like_parameter(
                    graft,
                    reference,
                )

                block.add_module(
                    cls.ATTNRES_NAMESPACE,
                    graft,
                )

        # ----------------------------------------------------
        # 添加 MoR modules
        # ----------------------------------------------------

        for block_index in sorted(
            mor_set
        ):
            block = model.blocks[
                block_index
            ]

            if hasattr(
                block,
                cls.MOR_NAMESPACE,
            ):
                existing = getattr(
                    block,
                    cls.MOR_NAMESPACE,
                )

                if not isinstance(
                    existing,
                    MoRGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在不兼容的 "
                        f"{cls.MOR_NAMESPACE}："
                        f"{type(existing).__name__}"
                    )
            else:
                graft = MoRGraft(
                    model_channels=model_channels,
                    context_dim=context_dim,
                    config=mor_config,
                    block_index=block_index,
                )

                reference = (
                    _find_block_reference_parameter(
                        block,
                        global_reference,
                    )
                )

                _move_module_like_parameter(
                    graft,
                    reference,
                )

                block.add_module(
                    cls.MOR_NAMESPACE,
                    graft,
                )

        # ----------------------------------------------------
        # 安装统一并行 block scheduler
        # ----------------------------------------------------

        blocks_to_wrap = (
            mudd_set
            | attnres_set
            | mor_set
            | set(model.extra_block_insert_after)
            | {0}
        )

        for block_index in sorted(
            blocks_to_wrap
        ):
            block = model.blocks[
                block_index
            ]

            _install_grafted4_block_forward(
                block=block,
                model=model,
                block_index=block_index,
                apply_mudd=(
                    block_index
                    in mudd_set
                ),
                apply_attnres=(
                    block_index
                    in attnres_set
                ),
                apply_mor=(
                    block_index
                    in mor_set
                ),
            )

        _install_model_runtime_context_forward(
            model
        )

        # ----------------------------------------------------
        # 模型元数据
        # ----------------------------------------------------

        object.__setattr__(
            model,
            "mudd_config",
            mudd_config,
        )

        object.__setattr__(
            model,
            "mudd_block_indices",
            sorted(
                mudd_set
            ),
        )

        object.__setattr__(
            model,
            "attnres_config",
            attnres_config,
        )

        object.__setattr__(
            model,
            "attnres_block_indices",
            sorted(
                attnres_set
            ),
        )

        object.__setattr__(
            model,
            "mor_config",
            mor_config,
        )

        object.__setattr__(
            model,
            "mor_block_indices",
            sorted(
                mor_set
            ),
        )

        object.__setattr__(
            model,
            "extra_block_config",
            extra_block_config,
        )

        object.__setattr__(
            model,
            "grafted4_strict_image_only",
            bool(
                attnres_config.strict_image_only
                and mor_config.strict_image_only
            ),
        )

        object.__setattr__(
            model,
            "graft_block_indices",
            sorted(
                mudd_set
                | attnres_set
                | mor_set
            ),
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

        # ----------------------------------------------------
        # 安装日志
        # ----------------------------------------------------

        print(
            "✅ [SpatialGraftV4] 已安装 hybrid image graft"
        )

        if extra_block_count:
            print(
                "✅ [SpatialGraftV4] 已恢复 residual extra blocks: "
                f"count={extra_block_count}, "
                f"insert_after={model.extra_block_insert_after}"
            )

        print(
            "ℹ️ [SpatialGraftV4] block placement: "
            f"MUDD={sorted(mudd_set)}, "
            f"AttnRes={sorted(attnres_set)}, "
            f"MoR={sorted(mor_set)}"
        )

        print(
            "ℹ️ [SpatialGraftV4] MUDD config: "
            f"memory_channels="
            f"{getattr(mudd_config, 'memory_channels_override', None)}, "
            f"memory_pool={mudd_config.memory_pool}, "
            f"memory_depth={mudd_config.memory_depth}, "
            f"detail={mudd_config.use_detail_branch}, "
            f"detail_channels="
            f"{getattr(mudd_config, 'detail_channels_override', None)}, "
            f"spatial_kernel={mudd_config.spatial_kernel_size}, "
            f"temporal_kernel={mudd_config.temporal_kernel_size}"
        )

        print(
            "ℹ️ [SpatialGraftV4] AttnRes config: "
            f"channels="
            f"{attnres_config.attention_channels_override}, "
            f"heads={attnres_config.num_heads}, "
            f"memory_pool={attnres_config.memory_pool}, "
            f"max_memory_tokens="
            f"{attnres_config.max_memory_tokens}, "
            f"memory_depth={attnres_config.memory_depth}, "
            f"prototypes="
            f"{attnres_config.num_prototype_tokens}, "
            f"spatial_kernel="
            f"{attnres_config.spatial_kernel_size}"
        )

        print(
            "ℹ️ [SpatialGraftV4] MoR config: "
            f"channels={mor_config.channels_override}, "
            f"heads={mor_config.num_heads}, "
            f"recursions={mor_config.max_recursions}, "
            f"memory_pool={mor_config.memory_pool}, "
            f"max_memory_tokens={mor_config.max_memory_tokens}, "
            f"memory_depth={mor_config.memory_depth}, "
            f"prototypes={mor_config.num_prototype_tokens}, "
            f"text={mor_config.use_text_conditioning}, "
            f"context_dim={context_dim}, "
            f"spatial_kernel={mor_config.spatial_kernel_size}, "
            f"dilation={mor_config.dilation}"
        )

        if global_reference is not None:
            print(
                "ℹ️ [SpatialGraftV4] 新增模块参考精度："
                f"dtype={global_reference.dtype}, "
                f"device={global_reference.device}"
            )

        overlap_mudd_attnres = sorted(
            mudd_set & attnres_set
        )

        overlap_mudd_mor = sorted(
            mudd_set & mor_set
        )

        overlap_attnres_mor = sorted(
            attnres_set & mor_set
        )

        if (
            overlap_mudd_attnres
            or overlap_mudd_mor
            or overlap_attnres_mor
        ):
            print(
                "ℹ️ [SpatialGraftV4] 并行重叠 blocks："
                f"MUDD/AttnRes={overlap_mudd_attnres}, "
                f"MUDD/MoR={overlap_mudd_mor}, "
                f"AttnRes/MoR={overlap_attnres_mor}"
            )

        return model


__all__ = [
    "GraftedAnima",

    "MUDDGraftRuntimeConfig",
    "AttnResRuntimeConfig",
    "MoRRuntimeConfig",

    "MUDDFormerGraft",
    "MUDDMemoryUnit",
    "MUDDDetailUnit",

    "SimpleRMSNorm",
    "ChannelNorm2d",
    "ImageMemoryUnit",
    "CompressedImageAttention",

    "ImageAttnResGraft",
    "AttnResGraft",

    "MoRSharedCell",
    "MoRGraft",

    "build_every_n_block_indices",
    "build_group_block_indices",
    "build_stride_block_indices",
]
