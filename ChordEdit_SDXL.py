from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib

import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.samplers

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED = 42


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    return max(multiple, (value // multiple) * multiple)


def _center_crop_bhwc(
        image: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    if image.ndim != 4:
        raise ValueError(
            "IMAGE 必须是 [B,H,W,C] 四维张量，"
            f"当前形状为 {tuple(image.shape)}。"
        )

    _, height, width, _ = image.shape

    if height == width:
        return image, 0, 0

    side = min(height, width)
    top = max(0, (height - side) // 2)
    left = max(0, (width - side) // 2)

    cropped = image[
        :,
        top:top + side,
        left:left + side,
        :,
    ]

    return cropped, left, top


def _resize_bhwc(
        image: torch.Tensor,
        width: int,
        height: int,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(
            "IMAGE 必须是 [B,H,W,C] 四维张量，"
            f"当前形状为 {tuple(image.shape)}。"
        )

    current_height = int(image.shape[1])
    current_width = int(image.shape[2])

    if current_width == int(width) and current_height == int(height):
        return image

    image_bchw = image.movedim(-1, 1)

    image_bchw = F.interpolate(
        image_bchw,
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    return image_bchw.movedim(1, -1)


@dataclass
class PreparedImage:
    image: torch.Tensor
    original_width: int
    original_height: int
    crop_x: int
    crop_y: int
    target_width: int
    target_height: int


def _prepare_image(
        image: torch.Tensor,
        width: int,
        height: int,
        center_crop: bool,
) -> PreparedImage:
    if not torch.is_tensor(image):
        raise TypeError("image 必须是 torch.Tensor。")

    if image.ndim != 4:
        raise ValueError(
            "期望 IMAGE 形状为 [B,H,W,C]，"
            f"但得到 {tuple(image.shape)}。"
        )

    if image.shape[-1] < 3:
        raise ValueError(
            "输入图片至少需要三个通道，"
            f"当前通道数为 {image.shape[-1]}。"
        )

    image = image[..., :3]
    image = image.detach().float().clamp(0.0, 1.0)

    input_height = int(image.shape[1])
    input_width = int(image.shape[2])

    crop_x = 0
    crop_y = 0

    if center_crop:
        image, crop_x, crop_y = _center_crop_bhwc(image)
        conditioned_height = int(image.shape[1])
        conditioned_width = int(image.shape[2])
    else:
        conditioned_height = input_height
        conditioned_width = input_width

    image = _resize_bhwc(
        image=image,
        width=width,
        height=height,
    ).clamp(0.0, 1.0)

    return PreparedImage(
        image=image,
        original_width=conditioned_width,
        original_height=conditioned_height,
        crop_x=crop_x,
        crop_y=crop_y,
        target_width=int(width),
        target_height=int(height),
    )


def _repeat_batch(
        tensor: torch.Tensor,
        batch_size: int,
) -> torch.Tensor:
    current_batch = int(tensor.shape[0])
    batch_size = int(batch_size)

    if current_batch == batch_size:
        return tensor

    if current_batch == 1:
        repeats = [batch_size] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)

    indices = (
            torch.arange(
                batch_size,
                device=tensor.device,
                dtype=torch.long,
            )
            % current_batch
    )

    return tensor.index_select(0, indices)


def _get_model_device(model: Any) -> torch.device:
    device = getattr(model, "load_device", None)

    if device is not None:
        return torch.device(device)

    try:
        return torch.device(model_management.get_torch_device())
    except Exception:
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


def _get_model_dtype(
        model: Any,
        fallback: torch.dtype = torch.float32,
) -> torch.dtype:
    candidates = [
        getattr(model, "model_dtype", None),
        getattr(getattr(model, "model", None), "dtype", None),
    ]

    inner_model = getattr(model, "model", None)

    if inner_model is not None:
        try:
            candidates.append(inner_model.get_dtype())
        except Exception:
            pass

    try:
        candidates.append(model.get_model_object("model_dtype"))
    except Exception:
        pass

    for candidate in candidates:
        if isinstance(candidate, torch.dtype):
            return candidate

        if callable(candidate):
            try:
                result = candidate()
                if isinstance(result, torch.dtype):
                    return result
            except Exception:
                pass

    return fallback


def _extract_latent_samples(latent: Any) -> torch.Tensor:
    if torch.is_tensor(latent):
        return latent

    if isinstance(latent, dict):
        samples = latent.get("samples")
        if torch.is_tensor(samples):
            return samples

    raise RuntimeError("无法从 VAE 编码结果中提取 latent tensor。")


def _tensor_stats(name: str, tensor: torch.Tensor) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return

    value = tensor.detach().float()

    LOGGER.info(
        "%s: shape=%s, mean_abs=%.8f, rms=%.8f, min=%.8f, max=%.8f",
        name,
        tuple(value.shape),
        float(value.abs().mean().cpu().item()),
        float(value.square().mean().sqrt().cpu().item()),
        float(value.min().cpu().item()),
        float(value.max().cpu().item()),
    )


def unwrap_comfy_condition(value: Any) -> Any:
    visited: set[int] = set()

    while value is not None and not torch.is_tensor(value):
        value_id = id(value)

        if value_id in visited:
            break

        visited.add(value_id)

        if hasattr(value, "cond"):
            value = value.cond
            continue

        break

    return value


def _get_comfy_api_function(name: str):
    module_names = (
        "comfy.samplers",
        "comfy.sampler_helpers",
    )

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        function = getattr(module, name, None)

        if callable(function):
            return function

    return None


def _convert_comfy_conditioning(
        conditioning: list,
) -> List[Dict[str, Any]]:
    if not isinstance(conditioning, list):
        raise TypeError(
            "待转换的 conditioning 必须是 list，"
            f"实际类型为 {type(conditioning).__name__}。"
        )

    if not conditioning:
        raise ValueError("待转换的 conditioning 不能为空。")

    convert_cond = _get_comfy_api_function("convert_cond")

    if not callable(convert_cond):
        raise RuntimeError(
            "当前 ComfyUI 找不到 convert_cond()。\n"
            "请确认 comfy.samplers / comfy.sampler_helpers 可正常导入。"
        )

    converted = convert_cond(conditioning)

    if not isinstance(converted, list):
        raise TypeError(
            "convert_cond() 返回值不是 list，"
            f"实际类型为 {type(converted).__name__}。"
        )

    if not converted:
        raise RuntimeError("convert_cond() 返回了空 conditioning。")

    for index, item in enumerate(converted):
        if not isinstance(item, dict):
            raise TypeError(
                "convert_cond() 转换后的条件条目不是 dict："
                f"index={index}, type={type(item).__name__}。"
            )

        if "model_conds" not in item:
            raise RuntimeError(
                "convert_cond() 转换后的条件缺少 model_conds："
                f"index={index}, keys={list(item.keys())}。"
            )

    return converted


def _encode_comfy_model_conds(
        model_extra_conds: Any,
        converted_conditioning: List[Dict[str, Any]],
        latent: torch.Tensor,
) -> List[Dict[str, Any]]:
    encode_model_conds = _get_comfy_api_function(
        "encode_model_conds"
    )

    if not callable(encode_model_conds):
        raise RuntimeError(
            "当前 ComfyUI 找不到 encode_model_conds()。"
        )

    if not callable(model_extra_conds):
        raise TypeError(
            "model_extra_conds 必须是可调用对象，"
            "通常应为 model_patcher.model.extra_conds。"
        )

    device = latent.device

    try:
        encoded = encode_model_conds(
            model_extra_conds,
            converted_conditioning,
            latent,
            device,
            "positive",
        )
    except TypeError:

        encoded = encode_model_conds(
            model_extra_conds,
            converted_conditioning,
            latent,
            device,
            prompt_type="positive",
        )

    if not isinstance(encoded, list):
        raise TypeError(
            "encode_model_conds() 返回值不是 list，"
            f"实际类型：{type(encoded).__name__}"
        )

    if not encoded:
        raise RuntimeError(
            "encode_model_conds() 返回了空 conditioning。"
        )

    return encoded


@dataclass
class ComfySDXLConditioning:
    cross_attn: torch.Tensor
    pooled_output: torch.Tensor
    time_ids: torch.Tensor

    def to(
            self,
            device: torch.device,
            dtype: Optional[torch.dtype] = None,
    ) -> "ComfySDXLConditioning":
        if dtype is None:
            dtype = self.cross_attn.dtype

        return ComfySDXLConditioning(
            cross_attn=self.cross_attn.to(
                device=device,
                dtype=dtype,
            ),
            pooled_output=self.pooled_output.to(
                device=device,
                dtype=dtype,
            ),
            time_ids=self.time_ids.to(
                device=device,
                dtype=dtype,
            ),
        )


def _dict_first_tensor(
        data: Dict[str, Any],
        keys: Sequence[str],
) -> Optional[torch.Tensor]:
    for key in keys:
        value = data.get(key)

        if value is None:
            continue

        value = unwrap_comfy_condition(value)

        if torch.is_tensor(value):
            return value

    return None


def _encode_clip_conditioning(
        clip: Any,
        prompt: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if clip is None:
        raise ValueError("CLIP 输入不能为空。")

    tokens = clip.tokenize(prompt or "")

    if hasattr(clip, "encode_from_tokens_scheduled"):
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if not conditioning:
            raise RuntimeError("CLIP 没有返回任何 conditioning。")

        if len(conditioning) > 1:
            LOGGER.warning(
                "提示词产生了 %d 个 scheduled conditioning 条目。"
                "ChordEdit 是静态条件算法，当前使用第一个条目。",
                len(conditioning),
            )

        first = conditioning[0]

        if isinstance(first, (list, tuple)) and len(first) >= 2:
            cross_attn = first[0]
            metadata = first[1] or {}

            if not isinstance(metadata, dict):
                raise RuntimeError(
                    "CLIP conditioning 的 metadata 不是字典，"
                    f"实际类型为 {type(metadata).__name__}。"
                )

            metadata = dict(metadata)

        elif hasattr(first, "cond"):
            cross_attn = first.cond
            metadata = {}

            params = getattr(first, "params", None)

            if isinstance(params, dict):
                metadata.update(params)

            if hasattr(first, "pooled_output"):
                metadata["pooled_output"] = first.pooled_output

        else:
            raise RuntimeError(
                "无法识别 CLIP conditioning 返回格式："
                f"{type(first).__name__}。"
            )

        cross_attn = unwrap_comfy_condition(cross_attn)

        if "pooled_output" in metadata:
            metadata["pooled_output"] = unwrap_comfy_condition(
                metadata["pooled_output"]
            )

        if "pooled" in metadata:
            metadata["pooled"] = unwrap_comfy_condition(
                metadata["pooled"]
            )

        if not torch.is_tensor(cross_attn):
            raise RuntimeError(
                "CLIP conditioning 中没有 cross-attention tensor。"
            )

        return cross_attn, metadata

    if hasattr(clip, "encode_from_tokens"):
        try:
            encoded = clip.encode_from_tokens(
                tokens,
                return_pooled=True,
                return_dict=True,
            )
        except TypeError:
            encoded = clip.encode_from_tokens(
                tokens,
                return_pooled=True,
            )

        if isinstance(encoded, dict):
            cross_attn = _dict_first_tensor(
                encoded,
                ("cond", "cross_attn"),
            )

            pooled = _dict_first_tensor(
                encoded,
                ("pooled_output", "pooled"),
            )

            if not torch.is_tensor(cross_attn):
                raise RuntimeError(
                    "旧版 CLIP 接口未返回 cross_attn。"
                )

            return cross_attn, {
                "pooled_output": pooled,
            }

        if isinstance(encoded, (list, tuple)) and len(encoded) >= 2:
            cross_attn = unwrap_comfy_condition(encoded[0])
            pooled = unwrap_comfy_condition(encoded[1])

            if not torch.is_tensor(cross_attn):
                raise RuntimeError(
                    "旧版 CLIP 接口未返回 cross_attn tensor。"
                )

            return cross_attn, {
                "pooled_output": pooled,
            }

    raise RuntimeError(
        "当前 CLIP 对象不支持可识别的 ComfyUI 编码接口。"
    )


def _build_time_ids(
        batch_size: int,
        original_width: int,
        original_height: int,
        target_width: int,
        target_height: int,
        crop_x: int = 0,
        crop_y: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = [
        float(original_height),
        float(original_width),
        float(crop_y),
        float(crop_x),
        float(target_height),
        float(target_width),
    ]

    return torch.tensor(
        [values],
        device=device,
        dtype=dtype,
    ).repeat(int(batch_size), 1)


def _make_conditioning(
        clip: Any,
        prompt: str,
        batch_size: int,
        original_width: int,
        original_height: int,
        target_width: int,
        target_height: int,
        crop_x: int = 0,
        crop_y: int = 0,
) -> ComfySDXLConditioning:
    cross_attn, metadata = _encode_clip_conditioning(
        clip=clip,
        prompt=prompt,
    )

    pooled_output = metadata.get("pooled_output")

    if pooled_output is None:
        pooled_output = metadata.get("pooled")

    pooled_output = unwrap_comfy_condition(pooled_output)

    if pooled_output is None:
        raise RuntimeError(
            "CLIP 没有返回 pooled_output。\n"
            "请确认输入的是 SDXL Checkpoint Loader 输出的 CLIP。"
        )

    if not torch.is_tensor(pooled_output):
        raise RuntimeError(
            "CLIP pooled_output 不是 torch.Tensor，"
            f"实际类型为 {type(pooled_output).__name__}。"
        )

    cross_attn = _repeat_batch(cross_attn, batch_size)
    pooled_output = _repeat_batch(pooled_output, batch_size)

    time_ids = _build_time_ids(
        batch_size=batch_size,
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        crop_x=crop_x,
        crop_y=crop_y,
        device=cross_attn.device,
        dtype=cross_attn.dtype,
    )

    return ComfySDXLConditioning(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        time_ids=time_ids,
    )


def _concat_conditioning(
        conditionings: Sequence[ComfySDXLConditioning],
) -> ComfySDXLConditioning:
    if not conditionings:
        raise ValueError("conditionings 不能为空。")

    return ComfySDXLConditioning(
        cross_attn=torch.cat(
            [item.cross_attn for item in conditionings],
            dim=0,
        ),
        pooled_output=torch.cat(
            [item.pooled_output for item in conditionings],
            dim=0,
        ),
        time_ids=torch.cat(
            [item.time_ids for item in conditionings],
            dim=0,
        ),
    )


class ComfySDXLModelAdapter:
    def __init__(self, model: Any) -> None:
        if model is None:
            raise ValueError("MODEL 输入不能为空。")

        self.model_patcher = model
        self.base_model = model.model

        if self.base_model is None:
            raise TypeError("输入 MODEL 不包含底层 model 对象。")

        self.device = _get_model_device(model)
        self.dtype = _get_model_dtype(model)

        self.model_sampling = getattr(
            self.base_model,
            "model_sampling",
            None,
        )

        if self.model_sampling is None:
            try:
                self.model_sampling = model.get_model_object(
                    "model_sampling"
                )
            except Exception:
                self.model_sampling = None

        if self.model_sampling is None:
            raise RuntimeError(
                "无法从 MODEL 获取 model_sampling。"
            )

    def load(self) -> None:
        model_management.load_models_gpu(
            [self.model_patcher]
        )

        self.device = _get_model_device(self.model_patcher)

        self.dtype = _get_model_dtype(
            self.model_patcher,
            fallback=self.dtype,
        )

    def time_to_sigma(
            self,
            t_scalar: float,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        t_scalar = max(0.0, min(1.0, float(t_scalar)))
        percent = 1.0 - t_scalar

        sigma_value = self.model_sampling.percent_to_sigma(
            percent
        )

        if torch.is_tensor(sigma_value):
            sigma_float = float(
                sigma_value.detach()
                .float()
                .reshape(-1)[0]
                .cpu()
                .item()
            )
        else:
            sigma_float = float(sigma_value)

        if not torch.isfinite(
                torch.tensor(sigma_float, dtype=torch.float64)
        ):
            raise RuntimeError(
                f"MODEL 返回了无效 sigma：{sigma_float}"
            )

        return torch.full(
            (int(batch_size),),
            sigma_float,
            device=device,
            dtype=dtype,
        )

    def _activate_base_model(
            self,
            device: torch.device,
    ) -> Tuple[Any, bool]:

        patcher = self.model_patcher

        model_management.load_models_gpu([patcher])

        base_model = patcher.model
        current_patcher = getattr(base_model, "current_patcher", None)

        if current_patcher is patcher:
            return base_model, False

        if current_patcher is not None and current_patcher is not patcher:
            raise RuntimeError(
                "SDXL ChordEdit 无法激活 MODEL：底层模型当前正被另一个 "
                "ModelPatcher 使用。"
            )

        try:
            active_model = patcher.patch_model(device_to=device)
        except TypeError:
            active_model = patcher.patch_model(device=device)

        if getattr(active_model, "current_patcher", None) is not patcher:
            LOGGER.debug(
                "patch_model() 未自动设置 current_patcher，"
                "为 calc_cond_batch() 临时设置。"
            )
            active_model.current_patcher = patcher

        return active_model, True

    def _pack_to_comfy_conditioning(
            self,
            conditioning: ComfySDXLConditioning,
            batch_size: int,
    ) -> list:

        cross_attn = _repeat_batch(
            conditioning.cross_attn,
            batch_size,
        )

        pooled = _repeat_batch(
            conditioning.pooled_output,
            batch_size,
        )

        time_ids = _repeat_batch(
            conditioning.time_ids,
            batch_size,
        )

        if time_ids.ndim != 2 or time_ids.shape[1] < 6:
            raise ValueError(
                "SDXL time_ids 必须为 [B,6]，"
                f"当前形状为 {tuple(time_ids.shape)}。"
            )

        original_height = int(
            round(float(time_ids[0, 0].item()))
        )
        original_width = int(
            round(float(time_ids[0, 1].item()))
        )
        crop_y = int(
            round(float(time_ids[0, 2].item()))
        )
        crop_x = int(
            round(float(time_ids[0, 3].item()))
        )
        target_height = int(
            round(float(time_ids[0, 4].item()))
        )
        target_width = int(
            round(float(time_ids[0, 5].item()))
        )

        cond_dict = {
            "pooled_output": pooled,

            "width": original_width,
            "height": original_height,
            "crop_w": crop_x,
            "crop_h": crop_y,
            "target_width": target_width,
            "target_height": target_height,
        }

        return [[cross_attn, cond_dict]]

    @staticmethod
    def _prepare_sigma(
            sigma: torch.Tensor,
            noisy_latent: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(noisy_latent.shape[0])

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(
                sigma,
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            )
        else:
            sigma = sigma.to(
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            )

        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)

        elif sigma.numel() == 1:
            sigma = sigma.reshape(1).expand(batch_size)

        else:
            sigma = sigma.reshape(-1)

            if sigma.shape[0] != batch_size:
                raise ValueError(
                    "sigma 批次与 noisy_latent 批次不匹配："
                    f"sigma={tuple(sigma.shape)}, "
                    f"latent={tuple(noisy_latent.shape)}。"
                )

        return sigma

    @torch.no_grad()
    def predict_x0(
            self,
            noisy_latent: torch.Tensor,
            sigma: torch.Tensor,
            conditioning: ComfySDXLConditioning,
    ) -> torch.Tensor:
        if not torch.is_tensor(noisy_latent):
            raise TypeError("noisy_latent 必须是 torch.Tensor。")

        if noisy_latent.ndim != 4:
            raise ValueError(
                "noisy_latent 必须是 [B, C, H, W] 格式，"
                f"当前为 {tuple(noisy_latent.shape)}。"
            )

        batch_size = int(noisy_latent.shape[0])

        raw_cond = self._pack_to_comfy_conditioning(
            conditioning=conditioning,
            batch_size=batch_size,
        )

        converted_cond = _convert_comfy_conditioning(raw_cond)

        model_extra_conds = getattr(
            self.base_model,
            "extra_conds",
            None,
        )

        if not callable(model_extra_conds):
            raise RuntimeError(
                "当前 MODEL 不支持 extra_conds()。"
                "SDXL ChordEdit 需要 SDXL 类型的 MODEL。"
            )

        encoded_cond = _encode_comfy_model_conds(
            model_extra_conds=model_extra_conds,
            converted_conditioning=converted_cond,
            latent=noisy_latent,
        )

        sigma = self._prepare_sigma(
            sigma=sigma,
            noisy_latent=noisy_latent,
        )

        model_options = getattr(
            self.model_patcher,
            "model_options",
            {},
        )

        if isinstance(model_options, dict):
            model_options = model_options.copy()
        else:
            model_options = {}

        active_model, should_unpatch = self._activate_base_model(
            device=noisy_latent.device,
        )

        try:
            result = comfy.samplers.calc_cond_batch(
                active_model,
                [encoded_cond],
                noisy_latent,
                sigma,
                model_options,
            )
        finally:

            if should_unpatch:
                try:
                    offload_device = model_management.unet_offload_device()

                    self.model_patcher.unpatch_model(
                        device_to=offload_device,
                    )

                    if (
                            getattr(
                                active_model,
                                "current_patcher",
                                None,
                            )
                            is self.model_patcher
                    ):
                        active_model.current_patcher = None

                except Exception:
                    LOGGER.exception(
                        "释放 SDXL ChordEdit 临时模型 patch 状态失败。"
                    )

        if not isinstance(result, (list, tuple)) or len(result) == 0:
            raise RuntimeError(
                "ComfyUI calc_cond_batch() 没有返回有效结果。"
            )

        denoised = result[0]

        if not torch.is_tensor(denoised):
            raise RuntimeError(
                "calc_cond_batch() 返回的预测结果不是 Tensor，"
                f"实际类型：{type(denoised).__name__}。"
            )

        if denoised.shape != noisy_latent.shape:
            raise RuntimeError(
                "模型输出 latent 形状与输入不一致："
                f"output={tuple(denoised.shape)}，"
                f"input={tuple(noisy_latent.shape)}。"
            )

        return denoised


class ComfySDXLChordEditEngine:
    def __init__(
            self,
            model: Any,
            clip: Any,
            vae: Any,
    ) -> None:
        self.model = model
        self.clip = clip
        self.vae = vae
        self.adapter = ComfySDXLModelAdapter(model=model)

    def load_model(self) -> None:
        self.adapter.load()

    @torch.no_grad()
    def encode_image(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.vae.encode(image[..., :3])
        return _extract_latent_samples(encoded)

    @torch.no_grad()
    def decode_latent(
            self,
            latent: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self.vae.decode(latent)

        if not torch.is_tensor(decoded):
            raise RuntimeError(
                "VAE.decode() 没有返回 IMAGE tensor。"
            )

        return decoded.detach().float().cpu().clamp(0.0, 1.0)

    def _make_noise(
            self,
            latent: torch.Tensor,
            seed: int,
            noise_samples: int,
    ) -> List[torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        noises: List[torch.Tensor] = []

        for _ in range(int(noise_samples)):
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(
                device=latent.device,
                dtype=latent.dtype,
            )

            noises.append(noise)

        return noises

    @torch.no_grad()
    def estimate_u(
            self,
            x_anchor: torch.Tensor,
            source_conditioning: ComfySDXLConditioning,
            target_conditioning: ComfySDXLConditioning,
            noises: Sequence[torch.Tensor],
            t_s: float,
            delta: float,
            unet_batch_size: int,
    ) -> torch.Tensor:
        if not noises:
            raise ValueError("至少需要一个噪声样本。")

        base_batch = int(x_anchor.shape[0])
        t_prev = max(0.0, float(t_s) - float(delta))

        sigma_s_base = self.adapter.time_to_sigma(
            t_scalar=t_s,
            batch_size=base_batch,
            device=x_anchor.device,
            dtype=x_anchor.dtype,
        )

        sigma_prev_base = self.adapter.time_to_sigma(
            t_scalar=t_prev,
            batch_size=base_batch,
            device=x_anchor.device,
            dtype=x_anchor.dtype,
        )

        sum_dv_s = torch.zeros_like(x_anchor)
        sum_dv_prev = torch.zeros_like(x_anchor)
        processed_noises = 0

        chunk_size = max(1, int(unet_batch_size))

        for chunk_start in range(
                0,
                len(noises),
                chunk_size,
        ):
            chunk = noises[
                chunk_start:chunk_start + chunk_size
            ]

            chunk_count = len(chunk)

            noise_stack = torch.cat(
                [
                    noise.to(
                        device=x_anchor.device,
                        dtype=x_anchor.dtype,
                    )
                    for noise in chunk
                ],
                dim=0,
            )

            anchor_stack = x_anchor.repeat(
                chunk_count,
                1,
                1,
                1,
            )

            sigma_s_stack = sigma_s_base.repeat(
                chunk_count
            )

            sigma_prev_stack = sigma_prev_base.repeat(
                chunk_count
            )

            sigma_s_stack_view = sigma_s_stack.view(
                chunk_count * base_batch,
                1,
                1,
                1,
            )

            sigma_prev_stack_view = sigma_prev_stack.view(
                chunk_count * base_batch,
                1,
                1,
                1,
            )

            z_s = (
                    anchor_stack
                    + sigma_s_stack_view * noise_stack
            )

            z_prev = (
                    anchor_stack
                    + sigma_prev_stack_view * noise_stack
            )

            model_input = torch.cat(
                [z_s, z_s, z_prev, z_prev],
                dim=0,
            )

            sigma_input = torch.cat(
                [
                    sigma_s_stack,
                    sigma_s_stack,
                    sigma_prev_stack,
                    sigma_prev_stack,
                ],
                dim=0,
            )

            source_chunk = ComfySDXLConditioning(
                cross_attn=source_conditioning.cross_attn.repeat(
                    chunk_count, 1, 1
                ),
                pooled_output=source_conditioning.pooled_output.repeat(
                    chunk_count, 1
                ),
                time_ids=source_conditioning.time_ids.repeat(
                    chunk_count, 1
                ),
            )

            target_chunk = ComfySDXLConditioning(
                cross_attn=target_conditioning.cross_attn.repeat(
                    chunk_count, 1, 1
                ),
                pooled_output=target_conditioning.pooled_output.repeat(
                    chunk_count, 1
                ),
                time_ids=target_conditioning.time_ids.repeat(
                    chunk_count, 1
                ),
            )

            all_conditioning = _concat_conditioning(
                [
                    source_chunk,
                    target_chunk,
                    source_chunk,
                    target_chunk,
                ]
            )

            x0_all = self.adapter.predict_x0(
                noisy_latent=model_input,
                sigma=sigma_input,
                conditioning=all_conditioning,
            )

            group_batch = chunk_count * base_batch

            x_source_s = x0_all[
                0 * group_batch:1 * group_batch
            ]

            x_target_s = x0_all[
                1 * group_batch:2 * group_batch
            ]

            x_source_prev = x0_all[
                2 * group_batch:3 * group_batch
            ]

            x_target_prev = x0_all[
                3 * group_batch:4 * group_batch
            ]

            dv_s = (
                    x_target_s - x_source_s
            ).reshape(
                chunk_count,
                base_batch,
                *x_anchor.shape[1:],
            ).sum(dim=0)

            dv_prev = (
                    x_target_prev - x_source_prev
            ).reshape(
                chunk_count,
                base_batch,
                *x_anchor.shape[1:],
            ).sum(dim=0)

            sum_dv_s.add_(dv_s)
            sum_dv_prev.add_(dv_prev)
            processed_noises += chunk_count

        dv_s_mean = sum_dv_s / float(processed_noises)
        dv_prev_mean = sum_dv_prev / float(processed_noises)

        _tensor_stats("ChordEdit dv_s", dv_s_mean)
        _tensor_stats("ChordEdit dv_prev", dv_prev_mean)

        denominator = float(t_s) + float(delta)

        if denominator <= 1e-6:
            return dv_s_mean

        return (
                float(delta) * dv_s_mean
                + float(t_s) * dv_prev_mean
        ) / denominator

    @torch.no_grad()
    def cleanup_prediction(
            self,
            x_current: torch.Tensor,
            target_conditioning: ComfySDXLConditioning,
            noise: torch.Tensor,
            t_end: float,
    ) -> torch.Tensor:
        batch_size = int(x_current.shape[0])

        sigma = self.adapter.time_to_sigma(
            t_scalar=t_end,
            batch_size=batch_size,
            device=x_current.device,
            dtype=x_current.dtype,
        )

        noisy_latent = (
                x_current
                + sigma.view(batch_size, 1, 1, 1) * noise
        )

        return self.adapter.predict_x0(
            noisy_latent=noisy_latent,
            sigma=sigma,
            conditioning=target_conditioning,
        )

    @torch.no_grad()
    def edit(
            self,
            image: torch.Tensor,
            source_prompt: str,
            target_prompt: str,
            seed: int,
            noise_samples: int,
            n_steps: int,
            t_start: float,
            t_end: float,
            t_delta: float,
            step_scale: float,
            direction_gain: float,
            cleanup: bool,
            target_width: int,
            target_height: int,
            center_crop: bool,
            force_square_when_cropping: bool,
            unet_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(
                "期望 IMAGE 为 [B,H,W,C]，"
                f"当前为 {tuple(image.shape)}。"
            )

        input_batch = int(image.shape[0])

        target_width = _round_to_multiple(
            target_width,
            8,
        )

        target_height = _round_to_multiple(
            target_height,
            8,
        )

        if center_crop and force_square_when_cropping:
            square_side = min(
                target_width,
                target_height,
            )

            target_width = square_side
            target_height = square_side

        noise_samples = max(1, int(noise_samples))
        n_steps = max(1, int(n_steps))

        t_start = max(0.0, min(1.0, float(t_start)))
        t_end = max(0.0, min(t_start, float(t_end)))
        t_delta = max(0.0, min(1.0, float(t_delta)))

        if t_start > 0.0 and t_delta >= t_start:
            t_delta = max(
                0.0,
                t_start - 1.0 / 999.0,
            )

        step_scale = float(step_scale)
        direction_gain = float(direction_gain)

        prepared = _prepare_image(
            image=image,
            width=target_width,
            height=target_height,
            center_crop=center_crop,
        )

        LOGGER.info(
            "使用 ComfyUI VAE 编码图像：%dx%d；"
            "SDXL original=%dx%d crop=(%d,%d)",
            target_width,
            target_height,
            prepared.original_width,
            prepared.original_height,
            prepared.crop_x,
            prepared.crop_y,
        )

        x_source = self.encode_image(
            prepared.image
        )

        self.load_model()

        model_device = self.adapter.device

        x_source = x_source.to(
            device=model_device
        )

        source_conditioning = _make_conditioning(
            clip=self.clip,
            prompt=source_prompt,
            batch_size=input_batch,
            original_width=prepared.original_width,
            original_height=prepared.original_height,
            target_width=prepared.target_width,
            target_height=prepared.target_height,
            crop_x=prepared.crop_x,
            crop_y=prepared.crop_y,
        )

        target_conditioning = _make_conditioning(
            clip=self.clip,
            prompt=target_prompt,
            batch_size=input_batch,
            original_width=prepared.original_width,
            original_height=prepared.original_height,
            target_width=prepared.target_width,
            target_height=prepared.target_height,
            crop_x=prepared.crop_x,
            crop_y=prepared.crop_y,
        )

        source_conditioning = source_conditioning.to(
            device=model_device,
            dtype=x_source.dtype,
        )

        target_conditioning = target_conditioning.to(
            device=model_device,
            dtype=x_source.dtype,
        )

        cross_difference = (
                target_conditioning.cross_attn
                - source_conditioning.cross_attn
        )

        pooled_difference = (
                target_conditioning.pooled_output
                - source_conditioning.pooled_output
        )

        _tensor_stats(
            "source/target cross_attn difference",
            cross_difference,
        )

        _tensor_stats(
            "source/target pooled difference",
            pooled_difference,
        )

        embedding_difference = float(
            cross_difference.detach()
            .float()
            .abs()
            .mean()
            .cpu()
            .item()
        )

        if embedding_difference < 1e-7:
            raise RuntimeError(
                "source_prompt 与 target_prompt 得到的 cross-attention "
                "几乎完全相同，无法生成编辑方向。\n"
                "请检查 CLIP 输入、提示词内容和模型类型。"
            )

        noises = self._make_noise(
            latent=x_source,
            seed=seed,
            noise_samples=noise_samples,
        )

        if n_steps == 1:
            time_grid = [float(t_start)]
        else:
            time_grid = torch.linspace(
                float(t_start),
                float(t_end),
                steps=n_steps,
                device="cpu",
                dtype=torch.float32,
            ).tolist()

        x_current = x_source.clone()

        for step_index, current_time in enumerate(time_grid):
            LOGGER.info(
                "SDXL ChordEdit：步骤 %d/%d，t=%.6f",
                step_index + 1,
                len(time_grid),
                float(current_time),
            )

            u_hat = self.estimate_u(
                x_anchor=x_current,
                source_conditioning=source_conditioning,
                target_conditioning=target_conditioning,
                noises=noises,
                t_s=float(current_time),
                delta=t_delta,
                unet_batch_size=unet_batch_size,
            )

            u_hat = u_hat * direction_gain
            update = step_scale * u_hat

            _tensor_stats("ChordEdit u_hat", u_hat)
            _tensor_stats("ChordEdit latent update", update)

            before = x_current
            x_current = x_current + update

            relative_change = (
                    update.detach().float().square().mean().sqrt()
                    / before.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                    .clamp_min(1e-8)
            )

            LOGGER.info(
                "ChordEdit 相对 latent 更新比例：%.8f",
                float(relative_change.cpu().item()),
            )

        total_change = x_current - x_source
        _tensor_stats("ChordEdit total latent change", total_change)

        if cleanup:
            LOGGER.info(
                "执行目标提示词 cleanup 预测，t_end=%.6f",
                t_end,
            )

            x_current = self.cleanup_prediction(
                x_current=x_current,
                target_conditioning=target_conditioning,
                noise=noises[0],
                t_end=t_end,
            )

        latent_output = x_current.detach()
        decoded = self.decode_latent(latent_output)

        return decoded, latent_output


class SDXLChordEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),

                "source_prompt": (
                    "STRING",
                    {
                        "default": "a photo of a person",
                        "multiline": True,
                    },
                ),

                "target_prompt": (
                    "STRING",
                    {
                        "default": "a watercolor painting of a person",
                        "multiline": True,
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": DEFAULT_SEED,
                        "min": 0,
                        "max": 0x7FFFFFFFFFFFFFFF,
                    },
                ),

                "noise_samples": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                    },
                ),

                "n_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),

                "t_start": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),

                "t_end": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),

                "t_delta": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                    },
                ),

                "step_scale": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),

                "direction_gain": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "放大 source/target 的编辑方向。"
                            "1.0 为原始公式；SDXL 建议从 2.0 开始。"
                        ),
                    },
                ),

                "cleanup": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),

                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 8,
                    },
                ),

                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 8,
                    },
                ),

                "center_crop": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),

                "force_square_when_cropping": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "启用中心裁剪时，强制使用正方形目标尺寸，"
                            "以匹配原始 ChordEdit 预处理。"
                        ),
                    },
                ),

                "unet_batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "一次批量处理多少个噪声样本。"
                            "每个噪声会生成四组模型输入。"
                        ),
                    },
                ),

                "clear_cache_after_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "run"
    CATEGORY = "image/editing/ChordEdit"

    DESCRIPTION = (
        "使用 ComfyUI MODEL、CLIP、VAE 执行 SDXL ChordEdit。"
    )

    @torch.no_grad()
    def run(
            self,
            model: Any,
            clip: Any,
            vae: Any,
            image: torch.Tensor,
            source_prompt: str,
            target_prompt: str,
            seed: int,
            noise_samples: int,
            n_steps: int,
            t_start: float,
            t_end: float,
            t_delta: float,
            step_scale: float,
            direction_gain: float,
            cleanup: bool,
            width: int,
            height: int,
            center_crop: bool,
            force_square_when_cropping: bool,
            unet_batch_size: int,
            clear_cache_after_run: bool,
    ):
        if model is None:
            raise ValueError("MODEL 输入不能为空。")

        if clip is None:
            raise ValueError("CLIP 输入不能为空。")

        if vae is None:
            raise ValueError("VAE 输入不能为空。")

        if not source_prompt.strip():
            LOGGER.warning(
                "source_prompt 为空，编辑方向可能不稳定。"
            )

        if not target_prompt.strip():
            LOGGER.warning(
                "target_prompt 为空，编辑方向可能不稳定。"
            )

        if source_prompt.strip() == target_prompt.strip():
            raise ValueError(
                "source_prompt 与 target_prompt 完全相同，"
                "编辑方向必然接近零。"
            )

        engine = ComfySDXLChordEditEngine(
            model=model,
            clip=clip,
            vae=vae,
        )

        try:
            output_image, output_latent = engine.edit(
                image=image,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                seed=int(seed),
                noise_samples=int(noise_samples),
                n_steps=int(n_steps),
                t_start=float(t_start),
                t_end=float(t_end),
                t_delta=float(t_delta),
                step_scale=float(step_scale),
                direction_gain=float(direction_gain),
                cleanup=bool(cleanup),
                target_width=int(width),
                target_height=int(height),
                center_crop=bool(center_crop),
                force_square_when_cropping=bool(
                    force_square_when_cropping
                ),
                unet_batch_size=int(unet_batch_size),
            )

            return (
                output_image,
                {
                    "samples": output_latent,
                },
            )

        finally:
            del engine

            if clear_cache_after_run:
                gc.collect()

                try:
                    model_management.soft_empty_cache()
                except TypeError:
                    try:
                        model_management.soft_empty_cache(
                            force=True
                        )
                    except Exception:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


class SDXLChordEditReleaseCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "release"
    CATEGORY = "image/editing/ChordEdit"

    def release(self, image: torch.Tensor):
        gc.collect()

        try:
            model_management.soft_empty_cache()
        except TypeError:
            try:
                model_management.soft_empty_cache(force=True)
            except Exception:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (image,)


NODE_CLASS_MAPPINGS = {
    "ComfySDXLChordEdit": SDXLChordEditNode,
    "ComfySDXLChordEditReleaseCache": SDXLChordEditReleaseCacheNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfySDXLChordEdit": "SDXL ChordEdit (MODEL/CLIP/VAE)",
    "ComfySDXLChordEditReleaseCache": (
        "SDXL ChordEdit Release Cache"
    ),
}
