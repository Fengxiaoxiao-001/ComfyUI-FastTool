# coding=utf-8
# Mutation/anima_spatial_graft_v1.py


import copy
import math
import re
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class SpatialGraftConfig:
    enabled: bool = True

    bottleneck_ratio: float = 0.25
    hidden_channels_override: Optional[int] = None
    conv_depth: int = 3

    temporal_kernel_size: int = 1
    dropout: float = 0.0

    branch_scale_init: float = 1.0
    spatial_layer_scale_init: float = 0.1
    temporal_layer_scale_init: float = 0.1

    use_framewise_timestep: bool = True
    use_rms_norm: bool = False




def _module_first_floating_parameter(
        module: Optional[nn.Module],
):
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


def _parameter_dtype_device(
        module: nn.Module,
        fallback_tensor: Optional[torch.Tensor] = None,
):
    parameter = None

    try:
        parameter = next(module.parameters())
    except StopIteration:
        pass

    if parameter is not None:
        return parameter.dtype, parameter.device

    if fallback_tensor is not None:
        return fallback_tensor.dtype, fallback_tensor.device

    return torch.float32, torch.device("cpu")


def _require_same_device(
        tensor: torch.Tensor,
        expected_device: torch.device,
        tensor_name: str,
):
    if tensor.device != expected_device:
        raise RuntimeError(
            f"[SpatialGraft] {tensor_name} 与模块不在同一设备："
            f"tensor={tensor.device}, module={expected_device}。\n"
            "这通常说明 ComfyUI 模型调度没有把新增 Mutation 模块"
            "移动到正确设备。"
        )


def _cast_tensor(
        tensor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    if tensor.device != device or tensor.dtype != dtype:
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
            f"[SpatialGraft] {tensor_name} 中检测到 NaN/Inf。"
            "已终止推理，避免输出纯黑图。"
        )




class ChannelLayerNorm2d(nn.Module):


    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.channels = int(channels)
        self.eps = float(eps)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = F.layer_norm(
            x.float(),
            normalized_shape=(self.channels,),
            weight=None,
            bias=None,
            eps=self.eps,
        )

        x = x.to(dtype=input_dtype)

        return x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()


class SimpleRMSNorm(nn.Module):


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
                torch.ones(self.dim)
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

        x_float = x.float()

        inverse_rms = torch.rsqrt(
            x_float.square().mean(
                dim=-1,
                keepdim=True,
            ) + self.eps
        )

        x = (
                x_float * inverse_rms
        ).to(dtype=input_dtype)

        if self.weight is not None:
            x = x * self.weight.to(
                device=x.device,
                dtype=input_dtype,
            )

        return x


class DepthwiseSpatialUnit(nn.Module):
    def __init__(
            self,
            channels: int,
            dropout: float = 0.0,
            layer_scale_init: float = 0.1,
    ):
        super().__init__()

        if layer_scale_init <= 0.0:
            raise ValueError(
                "layer_scale_init 必须大于 0"
            )

        channels = int(channels)

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.norm = ChannelLayerNorm2d(
            channels,
            eps=1e-6,
        )

        self.pointwise_in = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=1,
            bias=False,
        )

        self.activation = nn.GELU(
            approximate="tanh"
        )

        self.pointwise_out = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = (
            nn.Dropout2d(dropout)
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
            "DepthwiseSpatialUnit 输入",
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
        )

        result = residual + scale * branch


        result = _cast_tensor(
            result,
            dtype=compute_dtype,
            device=compute_device,
        )

        return result


class LightweightSpatialGraft(nn.Module):
    def __init__(
            self,
            model_channels: int,
            config: SpatialGraftConfig,
            block_index: int,
    ):
        super().__init__()

        self.model_channels = int(
            model_channels
        )

        self.block_index = int(
            block_index
        )

        self.config = copy.deepcopy(config)

        if (
                config.hidden_channels_override
                is not None
        ):
            hidden_channels = int(
                config.hidden_channels_override
            )

            if hidden_channels <= 0:
                raise ValueError(
                    "hidden_channels_override "
                    "必须大于 0"
                )
        else:
            hidden_channels = max(
                32,
                int(
                    self.model_channels
                    * config.bottleneck_ratio
                ),
            )

            hidden_channels = int(
                math.ceil(
                    hidden_channels / 32
                ) * 32
            )

            hidden_channels = min(
                hidden_channels,
                self.model_channels,
            )

        self.hidden_channels = int(
            hidden_channels
        )

        if config.use_rms_norm:
            self.input_norm = SimpleRMSNorm(
                self.model_channels,
                eps=1e-6,
                elementwise_affine=False,
            )
        else:
            self.input_norm = nn.LayerNorm(
                self.model_channels,
                elementwise_affine=False,
                eps=1e-6,
            )

        self.time_modulation = nn.Sequential(
            nn.SiLU(),

            nn.Linear(
                self.model_channels,
                self.hidden_channels,
                bias=False,
            ),

            nn.SiLU(),

            nn.Linear(
                self.hidden_channels,
                3 * self.model_channels,
                bias=True,
            ),
        )

        self.in_proj = nn.Conv2d(
            self.model_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.spatial_units = nn.ModuleList(
            [
                DepthwiseSpatialUnit(
                    self.hidden_channels,
                    dropout=config.dropout,
                    layer_scale_init=(
                        config.spatial_layer_scale_init
                    ),
                )
                for _ in range(
                int(config.conv_depth)
            )
            ]
        )

        if config.temporal_kernel_size > 1:
            kernel = int(
                config.temporal_kernel_size
            )

            if kernel % 2 != 1:
                raise ValueError(
                    "temporal_kernel_size 必须是奇数"
                )

            self.temporal_conv = nn.Conv3d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=(
                    kernel,
                    1,
                    1,
                ),
                stride=1,
                padding=(
                    kernel // 2,
                    0,
                    0,
                ),
                groups=self.hidden_channels,
                bias=False,
            )

            self.temporal_scale = nn.Parameter(
                torch.full(
                    (self.hidden_channels,),
                    float(
                        config.temporal_layer_scale_init
                    ),
                    dtype=torch.float32,
                )
            )
        else:
            self.temporal_conv = None

            self.register_parameter(
                "temporal_scale",
                None,
            )

        self.out_proj = nn.Conv2d(
            self.hidden_channels,
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
            self.in_proj.weight,
            a=math.sqrt(5),
        )

        for unit in self.spatial_units:
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

        if self.temporal_conv is not None:
            nn.init.kaiming_uniform_(
                self.temporal_conv.weight,
                a=math.sqrt(5),
            )


        nn.init.zeros_(
            self.out_proj.weight
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

    def _run_spatial_units(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        for unit in self.spatial_units:
            x = unit(x)

        return x

    def _apply_temporal_conv(
            self,
            x_btchw: torch.Tensor,
            batch_size: int,
            num_frames: int,
            compute_dtype: torch.dtype,
            compute_device: torch.device,
    ) -> torch.Tensor:
        if (
                self.temporal_conv is None
                or num_frames <= 1
        ):
            return x_btchw

        x_btchw = _cast_tensor(
            x_btchw,
            dtype=compute_dtype,
            device=compute_device,
        )

        batch_frames, channels, height, width = (
            x_btchw.shape
        )

        expected = batch_size * num_frames

        if batch_frames != expected:
            raise ValueError(
                "Temporal graft reshape 失败："
                f"BT={batch_frames}, "
                f"B={batch_size}, "
                f"T={num_frames}"
            )

        x_3d = x_btchw.reshape(
            batch_size,
            num_frames,
            channels,
            height,
            width,
        ).permute(
            0,
            2,
            1,
            3,
            4,
        ).contiguous()

        temporal_result = self.temporal_conv(
            x_3d
        )

        temporal_result = _cast_tensor(
            temporal_result,
            dtype=compute_dtype,
            device=compute_device,
        )

        temporal_scale = self.temporal_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        ).reshape(
            1,
            -1,
            1,
            1,
            1,
        )

        x_3d = (
                x_3d
                + temporal_scale * temporal_result
        )

        x_3d = _cast_tensor(
            x_3d,
            dtype=compute_dtype,
            device=compute_device,
        )

        return x_3d.permute(
            0,
            2,
            1,
            3,
            4,
        ).contiguous().reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )

    def forward(
            self,
            x_B_T_H_W_D: torch.Tensor,
            timestep_emb_B_T_D: torch.Tensor,
    ) -> torch.Tensor:
        if x_B_T_H_W_D.ndim != 5:
            raise ValueError(
                "Spatial graft 期望输入为 "
                "[B,T,H,W,D]，实际为 "
                f"{tuple(x_B_T_H_W_D.shape)}"
            )

        if not torch.is_tensor(
                timestep_emb_B_T_D
        ):
            raise TypeError(
                "Spatial graft timestep embedding "
                "必须是 Tensor"
            )

        if timestep_emb_B_T_D.ndim != 3:
            raise ValueError(
                "Spatial graft timestep embedding "
                "必须为 [B,T,D]，实际为 "
                f"{tuple(timestep_emb_B_T_D.shape)}"
            )

        compute_dtype = self.in_proj.weight.dtype
        compute_device = self.in_proj.weight.device

        _require_same_device(
            x_B_T_H_W_D,
            compute_device,
            "Spatial graft 特征输入",
        )

        _require_same_device(
            timestep_emb_B_T_D,
            compute_device,
            "Spatial graft timestep embedding",
        )


        x_B_T_H_W_D = _cast_tensor(
            x_B_T_H_W_D,
            dtype=compute_dtype,
            device=compute_device,
        )

        timestep_emb_B_T_D = _cast_tensor(
            timestep_emb_B_T_D,
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
                "Spatial graft 通道数不匹配："
                f"input={channels}, "
                f"expected={self.model_channels}"
            )

        if (
                timestep_emb_B_T_D.shape[0]
                != batch_size
        ):
            raise ValueError(
                "Spatial graft timestep batch "
                "与输入 batch 不一致"
            )

        if (
                timestep_emb_B_T_D.shape[-1]
                != channels
        ):
            raise ValueError(
                "Spatial graft timestep channel "
                "与输入 channel 不一致"
            )

        timestep_frames = (
            timestep_emb_B_T_D.shape[1]
        )

        if self.config.use_framewise_timestep:
            if (
                    timestep_frames == 1
                    and num_frames > 1
            ):
                timestep_emb_B_T_D = (
                    timestep_emb_B_T_D.expand(
                        batch_size,
                        num_frames,
                        channels,
                    )
                )
            elif timestep_frames != num_frames:
                raise ValueError(
                    "Spatial graft timestep 帧数"
                    "与 latent 帧数不一致："
                    f"{timestep_frames} != "
                    f"{num_frames}"
                )
        else:
            timestep_emb_B_T_D = (
                timestep_emb_B_T_D[
                    :,
                    :1,
                    :,
                ].expand(
                    batch_size,
                    num_frames,
                    channels,
                )
            )

        timestep_emb_B_T_D = (
            timestep_emb_B_T_D.contiguous()
        )

        time_parameter = (
            _module_first_floating_parameter(
                self.time_modulation
            )
        )

        if time_parameter is not None:
            timestep_emb_B_T_D = (
                _cast_tensor(
                    timestep_emb_B_T_D,
                    dtype=time_parameter.dtype,
                    device=time_parameter.device,
                )
            )

        modulation = self.time_modulation(
            timestep_emb_B_T_D
        )

        modulation = _cast_tensor(
            modulation,
            dtype=compute_dtype,
            device=compute_device,
        )

        shift, scale, timestep_gate = (
            modulation.chunk(
                3,
                dim=-1,
            )
        )

        shift = shift[
            :,
            :,
            None,
            None,
            :,
        ].to(dtype=compute_dtype)

        scale = scale[
            :,
            :,
            None,
            None,
            :,
        ].to(dtype=compute_dtype)

        timestep_gate = timestep_gate[
            :,
            :,
            None,
            None,
            :,
        ].to(dtype=compute_dtype)

        x = self.input_norm(
            x_B_T_H_W_D
        )

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        one = torch.ones(
            (),
            device=compute_device,
            dtype=compute_dtype,
        )

        x = (
                x * (one + scale)
                + shift
        )

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        x = x.permute(
            0,
            1,
            4,
            2,
            3,
        ).contiguous().reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )

        x = self.in_proj(x)

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        x = self._run_spatial_units(x)

        x = self._apply_temporal_conv(
            x,
            batch_size=batch_size,
            num_frames=num_frames,
            compute_dtype=compute_dtype,
            compute_device=compute_device,
        )

        x = self.out_proj(x)

        x = _cast_tensor(
            x,
            dtype=compute_dtype,
            device=compute_device,
        )

        x = x.reshape(
            batch_size,
            num_frames,
            channels,
            height,
            width,
        ).permute(
            0,
            1,
            3,
            4,
            2,
        ).contiguous()

        half = torch.tensor(
            0.5,
            device=compute_device,
            dtype=compute_dtype,
        )

        time_strength = (
                one
                + half * torch.tanh(
            timestep_gate
        )
        )

        time_strength = _cast_tensor(
            time_strength,
            dtype=compute_dtype,
            device=compute_device,
        )

        branch_scale = self.branch_scale.to(
            device=compute_device,
            dtype=compute_dtype,
        )

        result = (
                residual
                + branch_scale
                * time_strength
                * x
        )


        result = _cast_tensor(
            result,
            dtype=compute_dtype,
            device=compute_device,
        )

        return result




def build_front_dense_back_sparse_indices(
        num_blocks: int,
        dense_until_ratio: float = 0.45,
        dense_stride: int = 2,
        sparse_stride: int = 4,
        include_first: bool = False,
        include_last: bool = False,
) -> List[int]:
    if num_blocks <= 0:
        return []

    if not 0.0 < dense_until_ratio <= 1.0:
        raise ValueError(
            "dense_until_ratio 必须在 (0,1] 内"
        )

    if dense_stride <= 0:
        raise ValueError(
            "dense_stride 必须大于 0"
        )

    if sparse_stride <= 0:
        raise ValueError(
            "sparse_stride 必须大于 0"
        )

    dense_end = max(
        1,
        min(
            num_blocks,
            int(
                round(
                    num_blocks
                    * dense_until_ratio
                )
            ),
        ),
    )

    dense_start = (
        0
        if include_first
        else dense_stride
    )

    indices = list(
        range(
            dense_start,
            dense_end,
            dense_stride,
        )
    )

    sparse_start = dense_end

    if sparse_start % sparse_stride != 0:
        sparse_start += (
                sparse_stride
                - sparse_start % sparse_stride
        )

    indices.extend(
        range(
            sparse_start,
            num_blocks,
            sparse_stride,
        )
    )

    indices = sorted(
        {
            index
            for index in indices
            if 0 <= index < num_blocks
        }
    )

    if not include_first:
        indices = [
            index
            for index in indices
            if index != 0
        ]

    if not include_last:
        indices = [
            index
            for index in indices
            if index != num_blocks - 1
        ]

    return indices




def _install_grafted_block_forward(
        block: nn.Module,
):
    if getattr(
            block,
            "_spatial_graft_forward_installed",
            False,
    ):
        return

    if not hasattr(
            block,
            "spatial_graft",
    ):
        raise RuntimeError(
            "安装 grafted forward 之前，"
            "必须先添加 spatial_graft 模块"
        )

    original_forward = block.forward

    object.__setattr__(
        block,
        "_original_anima_forward",
        original_forward,
    )

    def grafted_forward(
            self,
            x_B_T_H_W_D: torch.Tensor,
            emb_B_T_D: torch.Tensor,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        x = self._original_anima_forward(
            x_B_T_H_W_D,
            emb_B_T_D,
            *args,
            **kwargs,
        )

        if not torch.is_tensor(x):
            raise TypeError(
                "[SpatialGraft] 原始 Anima block.forward "
                f"返回了 {type(x).__name__}，不是 Tensor。\n"
                "当前 Mutation 前向逻辑要求原始 block "
                "直接返回 [B,T,H,W,D] Tensor。"
            )

        return self.spatial_graft(
            x,
            emb_B_T_D,
        )

    block.forward = MethodType(
        grafted_forward,
        block,
    )

    object.__setattr__(
        block,
        "_spatial_graft_forward_installed",
        True,
    )




class GraftedAnima:


    MUTATION_API_VERSION = 1

    MUTATION_ID = "anima_spatial_graft_v1"

    DISPLAY_NAME = (
        "Anima Spatial Convolution Graft V1"
    )

    MODULE_NAMESPACE = "spatial_graft"

    DEFAULT_CONFIG = SpatialGraftConfig(
        enabled=True,
        bottleneck_ratio=0.25,
        hidden_channels_override=None,
        conv_depth=3,
        temporal_kernel_size=1,
        dropout=0.0,
        branch_scale_init=1.0,
        spatial_layer_scale_init=0.1,
        temporal_layer_scale_init=0.1,
        use_framewise_timestep=True,
        use_rms_norm=False,
    )

    @classmethod
    def detect(
            cls,
            state_dict_keys,
    ) -> int:
        score = 0

        for key in state_dict_keys:
            key_string = str(key)
            key_lower = key_string.lower()

            if cls.MUTATION_ID in key_lower:
                score = max(score, 120)

            elif ".spatial_graft." in key_lower:
                score = max(score, 100)

            elif "spatial_graft." in key_lower:
                score = max(score, 95)

            elif "spatial_graft_" in key_lower:
                score = max(score, 85)

            elif "spatial_graft" in key_lower:
                score = max(score, 80)

        return score

    @classmethod
    def is_mutation_key(
            cls,
            key,
    ) -> bool:
        key_lower = str(key).lower()

        return (
                ".spatial_graft." in key_lower
                or "spatial_graft." in key_lower
                or "spatial_graft_" in key_lower
                or cls.MUTATION_ID in key_lower
        )

    @classmethod
    def _infer_indices_from_keys(
            cls,
            source_keys,
    ) -> List[int]:
        indices = set()

        patterns = (
            re.compile(
                r"(?:^|\.)blocks\.(\d+)"
                r"\.spatial_graft(?:\.|$)",
                re.IGNORECASE,
            ),

            re.compile(
                r"(?:^|_)blocks_(\d+)"
                r"_spatial_graft(?:_|$)",
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
                        int(match.group(1))
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
    ) -> SpatialGraftConfig:
        config = copy.deepcopy(
            cls.DEFAULT_CONFIG
        )

        if not source_state_dict:
            return config

        hidden_channels = set()
        unit_indices = set()
        temporal_kernel_sizes = set()

        in_proj_pattern = re.compile(
            r"(?:^|\.)blocks\.(\d+)"
            r"\.spatial_graft\.in_proj\.weight$",
            re.IGNORECASE,
        )

        unit_pattern = re.compile(
            r"\.spatial_graft\.spatial_units\."
            r"(\d+)\.",
            re.IGNORECASE,
        )

        temporal_pattern = re.compile(
            r"\.spatial_graft\.temporal_conv\.weight$",
            re.IGNORECASE,
        )

        for key, tensor in source_state_dict.items():
            key_string = str(key)

            if not torch.is_tensor(tensor):
                continue

            if in_proj_pattern.search(key_string):
                if tensor.ndim != 4:
                    raise RuntimeError(
                        "Spatial Graft in_proj.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                hidden_channels.add(
                    int(tensor.shape[0])
                )

            unit_match = unit_pattern.search(
                key_string
            )

            if unit_match is not None:
                unit_indices.add(
                    int(unit_match.group(1))
                )

            if temporal_pattern.search(key_string):
                if tensor.ndim != 5:
                    raise RuntimeError(
                        "Spatial Graft temporal_conv.weight "
                        "维度异常："
                        f"{key_string} -> "
                        f"{tuple(tensor.shape)}"
                    )

                temporal_kernel_sizes.add(
                    int(tensor.shape[2])
                )

        if len(hidden_channels) > 1:
            raise RuntimeError(
                "同一个 Spatial Graft checkpoint "
                "包含多个 hidden_channels："
                f"{sorted(hidden_channels)}"
            )

        if hidden_channels:
            hidden = next(
                iter(hidden_channels)
            )

            config.hidden_channels_override = (
                hidden
            )

            config.bottleneck_ratio = (
                    float(hidden)
                    / float(model_channels)
            )

        if unit_indices:
            config.conv_depth = (
                    max(unit_indices) + 1
            )

        if len(temporal_kernel_sizes) > 1:
            raise RuntimeError(
                "同一个 Spatial Graft checkpoint "
                "包含多个 temporal kernel："
                f"{sorted(temporal_kernel_sizes)}"
            )

        if temporal_kernel_sizes:
            config.temporal_kernel_size = next(
                iter(temporal_kernel_sizes)
            )
        else:
            config.temporal_kernel_size = 1

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
                f"{existing_mutation_id}，"
                f"不能再次安装 {cls.MUTATION_ID}"
            )

        if not hasattr(model, "blocks"):
            raise AttributeError(
                "Anima diffusion_model "
                "不存在 blocks 属性"
            )

        if not hasattr(
                model,
                "model_channels",
        ):
            raise AttributeError(
                "Anima diffusion_model "
                "不存在 model_channels 属性"
            )

        num_blocks = len(model.blocks)

        all_source_keys = list(
            source_keys or []
        )

        if source_state_dict:
            all_source_keys.extend(
                source_state_dict.keys()
            )

        graft_block_indices = (
            cls._infer_indices_from_keys(
                all_source_keys
            )
        )

        if not graft_block_indices:
            graft_block_indices = (
                build_front_dense_back_sparse_indices(
                    num_blocks=num_blocks,
                    dense_until_ratio=0.45,
                    dense_stride=2,
                    sparse_stride=4,
                    include_first=False,
                    include_last=False,
                )
            )

        runtime_config = (
            cls._infer_config_from_state_dict(
                source_state_dict,
                model_channels=int(
                    model.model_channels
                ),
            )
        )

        reference_parameter = None


        x_embedder = getattr(
            model,
            "x_embedder",
            None,
        )

        if x_embedder is not None:
            try:
                for parameter in (
                        x_embedder.parameters()
                ):
                    if (
                            torch.is_tensor(parameter)
                            and parameter
                            .is_floating_point()
                    ):
                        reference_parameter = (
                            parameter
                        )
                        break
            except Exception:
                reference_parameter = None

        if reference_parameter is None:
            try:
                for parameter in (
                        model.parameters()
                ):
                    if (
                            torch.is_tensor(parameter)
                            and parameter
                            .is_floating_point()
                    ):
                        reference_parameter = (
                            parameter
                        )
                        break
            except Exception:
                reference_parameter = None

        for block_index in graft_block_indices:
            if not (
                    0
                    <= block_index
                    < num_blocks
            ):
                raise ValueError(
                    "无效的 Spatial Graft block："
                    f"{block_index}，"
                    f"总 block 数={num_blocks}"
                )

            block = model.blocks[
                block_index
            ]

            if hasattr(
                    block,
                    "spatial_graft",
            ):
                graft = block.spatial_graft

                if not isinstance(
                        graft,
                        LightweightSpatialGraft,
                ):
                    raise TypeError(
                        f"Block {block_index} 已存在"
                        "不兼容的 spatial_graft："
                        f"{type(graft).__name__}"
                    )
            else:
                graft = LightweightSpatialGraft(
                    model_channels=int(
                        model.model_channels
                    ),
                    config=runtime_config,
                    block_index=block_index,
                )

                if reference_parameter is not None:
                    graft.to(
                        device=(
                            reference_parameter.device
                        ),
                        dtype=(
                            reference_parameter.dtype
                        ),
                    )

                block.add_module(
                    "spatial_graft",
                    graft,
                )

            _install_grafted_block_forward(
                block
            )

        object.__setattr__(
            model,
            "graft_config",
            runtime_config,
        )

        object.__setattr__(
            model,
            "graft_block_indices",
            graft_block_indices,
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
            "✅ [SpatialGraftV1] 已安装到 blocks: "
            f"{graft_block_indices}"
        )

        print(
            "ℹ️ [SpatialGraftV1] 运行配置: "
            f"model_channels={model.model_channels}, "
            f"hidden_channels="
            f"{runtime_config.hidden_channels_override}, "
            f"conv_depth={runtime_config.conv_depth}, "
            f"temporal_kernel="
            f"{runtime_config.temporal_kernel_size}"
        )

        return model
