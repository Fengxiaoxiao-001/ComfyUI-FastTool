import torch

import folder_paths
import comfy.lora
import comfy.model_management
import comfy.sd
import comfy.supported_models_base
import comfy.utils

from comfy.sd import CLIP

try:
    from comfy.text_encoders.anima import (
        AnimaTEModel,
        AnimaTokenizer,
    )
except Exception:
    AnimaTEModel = None
    AnimaTokenizer = None

import types




def _dtype_from_string(
        dtype_name,
        fallback=None,
):
    dtype_map = {
        "auto": fallback,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return dtype_map.get(
        dtype_name,
        fallback,
    )


def _dtype_name(dtype):
    if dtype is None:
        return "None"

    return str(dtype).replace(
        "torch.",
        "",
    )


def _device_from_string(device):
    if device == "auto":
        return (
            comfy.model_management
            .get_torch_device()
        )

    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "⚠️ [AnimaBaker] CUDA 不可用，"
                "回退到 CPU"
            )

            return torch.device("cpu")

        return (
            comfy.model_management
            .get_torch_device()
        )

    if device == "npu":
        try:
            if (
                    hasattr(torch, "npu")
                    and torch.npu.is_available()
            ):
                return torch.device("npu")
        except Exception:
            pass

        print(
            "⚠️ [AnimaBaker] NPU 不可用，"
            "回退到自动设备"
        )

        return (
            comfy.model_management
            .get_torch_device()
        )

    return torch.device(device)


def _module_first_floating_parameter(
        module,
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


def _module_dtype(
        module,
        fallback=torch.bfloat16,
):
    parameter = (
        _module_first_floating_parameter(
            module
        )
    )

    if parameter is None:
        return fallback

    return parameter.dtype


def _module_device(
        module,
        fallback=torch.device("cpu"),
):
    parameter = (
        _module_first_floating_parameter(
            module
        )
    )

    if parameter is None:
        return fallback

    return parameter.device




_KNOWN_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _iter_module_floating_parameters(
        module,
):


    if module is None:
        return

    seen = set()

    try:
        parameters = module.parameters(
            recurse=True
        )
    except Exception:
        return

    for parameter in parameters:
        if not torch.is_tensor(parameter):
            continue

        if not parameter.is_floating_point():
            continue

        parameter_id = id(parameter)

        if parameter_id in seen:
            continue

        seen.add(parameter_id)

        yield parameter


def _module_floating_dtype_counts(
        module,
):


    result = {}

    for parameter in (
            _iter_module_floating_parameters(
                module
            )
    ):
        result[parameter.dtype] = (
                result.get(parameter.dtype, 0)
                + parameter.numel()
        )

    return result


def _format_dtype_counts(
        dtype_counts,
):
    if not dtype_counts:
        return "无浮点参数"

    items = sorted(
        dtype_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return ", ".join(
        f"{_dtype_name(dtype)}={count:,}"
        for dtype, count in items
    )


def _module_needs_dtype_normalization(
        module,
        target_dtype,
):


    if module is None:
        return False

    if target_dtype is None:
        return False

    found_parameter = False

    for parameter in (
            _iter_module_floating_parameters(
                module
            )
    ):
        found_parameter = True

        if parameter.dtype != target_dtype:
            return True

    return False if found_parameter else False


def _read_runtime_dtype_candidate(
        obj,
        attribute_name,
):
    if obj is None:
        return None

    try:
        value = getattr(
            obj,
            attribute_name,
        )
    except Exception:
        return None

    if callable(value):
        try:
            value = value()
        except Exception:
            return None

    if value in _KNOWN_FLOAT_DTYPES:
        return value

    return None


def _resolve_model_runtime_dtype(
        patcher,
        module,
        fallback,
):


    objects = []

    patcher_model = getattr(
        patcher,
        "model",
        None,
    )

    objects.extend(
        [
            patcher_model,
            patcher,
            module,
        ]
    )

    if patcher_model is not None:
        objects.extend(
            [
                getattr(
                    patcher_model,
                    "diffusion_model",
                    None,
                ),
                getattr(
                    patcher_model,
                    "model",
                    None,
                ),
            ]
        )

    objects.append(
        getattr(
            module,
            "diffusion_model",
            None,
        )
    )

    attribute_names = (
        "model_dtype",
        "get_dtype",
        "dtype",
    )

    seen = set()

    for obj in objects:
        if obj is None:
            continue

        obj_id = id(obj)

        if obj_id in seen:
            continue

        seen.add(obj_id)

        for attribute_name in attribute_names:
            candidate = (
                _read_runtime_dtype_candidate(
                    obj,
                    attribute_name,
                )
            )

            if candidate is not None:
                return candidate

    return fallback


def _collect_related_runtime_objects(
        patcher,
        module,
):

    result = []
    queue = [
        patcher,
        getattr(patcher, "model", None),
        module,
    ]

    seen = set()

    while queue:
        obj = queue.pop(0)

        if obj is None:
            continue

        obj_id = id(obj)

        if obj_id in seen:
            continue

        seen.add(obj_id)
        result.append(obj)

        for attribute_name in (
                "model",
                "diffusion_model",
        ):
            try:
                child = getattr(
                    obj,
                    attribute_name,
                    None,
                )
            except Exception:
                child = None

            if (
                    child is not None
                    and child is not obj
            ):
                queue.append(child)

    return result


def _set_runtime_dtype_metadata(
        patcher,
        module,
        target_dtype,
):


    if target_dtype not in _KNOWN_FLOAT_DTYPES:
        return

    for obj in _collect_related_runtime_objects(
            patcher,
            module,
    ):
        try:
            current_dtype = getattr(
                obj,
                "dtype",
                None,
            )
        except Exception:
            current_dtype = None

        if current_dtype in _KNOWN_FLOAT_DTYPES:
            try:
                setattr(
                    obj,
                    "dtype",
                    target_dtype,
                )
            except Exception:
                pass


def _find_x_embedder_parameter(
        diffusion_model,
):
    if diffusion_model is None:
        return None

    x_embedder = getattr(
        diffusion_model,
        "x_embedder",
        None,
    )

    parameter = (
        _module_first_floating_parameter(
            x_embedder
        )
    )

    if parameter is not None:
        return parameter

    return _module_first_floating_parameter(
        diffusion_model
    )


def _patch_anima_runtime_inputs(
        diffusion_model,
):


    if diffusion_model is None:
        return

    if getattr(
            diffusion_model,
            "_anima_runtime_input_dtype_patched",
            False,
    ):
        return

    original_forward = (
        diffusion_model.forward
    )

    def patched_forward(
            self,
            x,
            *args,
            **kwargs,
    ):
        reference_parameter = (
            _find_x_embedder_parameter(
                self
            )
        )

        if reference_parameter is None:
            return original_forward(
                x,
                *args,
                **kwargs,
            )

        target_dtype = (
            reference_parameter.dtype
        )

        target_device = (
            reference_parameter.device
        )

        if (
                torch.is_tensor(x)
                and x.is_floating_point()
                and (
                x.dtype != target_dtype
                or x.device != target_device
        )
        ):
            x = x.to(
                device=target_device,
                dtype=target_dtype,
            )


        positional_args = list(args)

        if (
                len(positional_args) >= 2
                and torch.is_tensor(
            positional_args[1]
        )
                and positional_args[1]
                .is_floating_point()
        ):
            context = positional_args[1]

            if (
                    context.dtype != target_dtype
                    or context.device
                    != target_device
            ):
                positional_args[1] = (
                    context.to(
                        device=target_device,
                        dtype=target_dtype,
                    )
                )

        if (
                "context" in kwargs
                and torch.is_tensor(
            kwargs["context"]
        )
                and kwargs["context"]
                .is_floating_point()
        ):
            context = kwargs["context"]

            if (
                    context.dtype != target_dtype
                    or context.device
                    != target_device
            ):
                kwargs["context"] = (
                    context.to(
                        device=target_device,
                        dtype=target_dtype,
                    )
                )

        return original_forward(
            x,
            *positional_args,
            **kwargs,
        )

    diffusion_model.forward = (
        types.MethodType(
            patched_forward,
            diffusion_model,
        )
    )

    diffusion_model._anima_runtime_input_dtype_patched = (
        True
    )

    print(
        "✅ [AnimaBaker] 已安装 Anima 主输入 "
        "dtype/device 防护"
    )


def _find_model_path(
        model_name,
        folder_types,
):
    for folder_type in folder_types:
        path = folder_paths.get_full_path(
            folder_type,
            model_name,
        )

        if path is not None:
            return path, folder_type

    return None, None


def _find_diffusion_model_from_patcher(
        model_patcher,
):
    candidates = []

    if hasattr(model_patcher, "model"):
        candidates.append(
            model_patcher.model
        )

        if hasattr(
                model_patcher.model,
                "diffusion_model",
        ):
            candidates.append(
                model_patcher
                .model
                .diffusion_model
            )

        if hasattr(
                model_patcher.model,
                "model",
        ):
            candidates.append(
                model_patcher
                .model
                .model
            )

            if hasattr(
                    model_patcher
                            .model
                            .model,
                    "diffusion_model",
            ):
                candidates.append(
                    model_patcher
                    .model
                    .model
                    .diffusion_model
                )

    if hasattr(
            model_patcher,
            "diffusion_model",
    ):
        candidates.append(
            model_patcher.diffusion_model
        )

    candidates.append(
        model_patcher
    )

    for obj in candidates:
        if (
                obj is not None
                and hasattr(obj, "blocks")
                and hasattr(
            obj,
            "model_channels",
        )
        ):
            return obj

    for obj in candidates:
        if (
                obj is not None
                and hasattr(
            obj,
            "preprocess_text_embeds",
        )
        ):
            return obj

    raise RuntimeError(
        "找不到 Anima diffusion_model"
    )


def _get_clip_patcher(
        clip_obj,
):
    if clip_obj is None:
        return None

    patcher = getattr(
        clip_obj,
        "patcher",
        None,
    )

    return patcher


def _patcher_has_weight_patches(
        patcher,
):
    if patcher is None:
        return False

    patches = getattr(
        patcher,
        "patches",
        None,
    )

    return bool(patches)


def _force_anima_llm_adapter_device(
        diffusion_model,
        device=None,
        dtype=None,
):


    if (
            diffusion_model is None
            or not hasattr(
        diffusion_model,
        "llm_adapter",
    )
    ):
        return

    llm_adapter = (
        diffusion_model.llm_adapter
    )

    model_parameter = (
        _module_first_floating_parameter(
            diffusion_model
        )
    )

    if device is None:
        if model_parameter is not None:
            device = model_parameter.device
        else:
            device = torch.device("cpu")

    if dtype is None:
        if model_parameter is not None:
            dtype = model_parameter.dtype
        else:
            dtype = torch.bfloat16

    try:
        llm_adapter.to(
            device=device,
            dtype=dtype,
        )
    except TypeError:
        llm_adapter.to(
            device=device
        )


    try:
        embed = getattr(
            llm_adapter,
            "embed",
            None,
        )

        if embed is not None:
            embed.to(
                device=device
            )
    except Exception:
        pass


def _patch_anima_preprocess_text_embeds(
        diffusion_model,
):


    if diffusion_model is None:
        return

    if not hasattr(
            diffusion_model,
            "preprocess_text_embeds",
    ):
        return

    if getattr(
            diffusion_model,
            "_anima_device_preprocess_patched",
            False,
    ):
        return

    original_preprocess = (
        diffusion_model
        .preprocess_text_embeds
    )

    def patched_preprocess(
            self,
            text_embeds,
            text_ids,
            *args,
            **kwargs,
    ):
        adapter = getattr(
            self,
            "llm_adapter",
            None,
        )

        reference_parameter = (
            _module_first_floating_parameter(
                adapter
            )
        )

        if reference_parameter is None:
            reference_parameter = (
                _module_first_floating_parameter(
                    self
                )
            )

        if reference_parameter is not None:
            target_dtype = (
                reference_parameter.dtype
            )

            target_device = (
                reference_parameter.device
            )
        else:
            target_dtype = (
                text_embeds.dtype
                if (
                        torch.is_tensor(text_embeds)
                        and text_embeds
                        .is_floating_point()
                )
                else torch.bfloat16
            )

            target_device = (
                text_embeds.device
                if torch.is_tensor(text_embeds)
                else torch.device("cpu")
            )

        if (
                torch.is_tensor(text_embeds)
                and text_embeds.is_floating_point()
                and (
                text_embeds.device
                != target_device
                or text_embeds.dtype
                != target_dtype
        )
        ):
            text_embeds = text_embeds.to(
                device=target_device,
                dtype=target_dtype,
            )

        if torch.is_tensor(text_ids):
            if text_ids.device != target_device:
                text_ids = text_ids.to(
                    device=target_device
                )

        if (
                "t5xxl_weights" in kwargs
                and torch.is_tensor(
            kwargs["t5xxl_weights"]
        )
        ):
            t5xxl_weights = (
                kwargs["t5xxl_weights"]
            )

            if t5xxl_weights.is_floating_point():
                kwargs["t5xxl_weights"] = (
                    t5xxl_weights.to(
                        device=target_device,
                        dtype=target_dtype,
                    )
                )
            else:
                kwargs["t5xxl_weights"] = (
                    t5xxl_weights.to(
                        device=target_device
                    )
                )

        return original_preprocess(
            text_embeds,
            text_ids,
            *args,
            **kwargs,
        )

    diffusion_model.preprocess_text_embeds = (
        types.MethodType(
            patched_preprocess,
            diffusion_model,
        )
    )

    diffusion_model._anima_device_preprocess_patched = (
        True
    )

    print(
        "✅ [AnimaBaker] 已修复 Anima "
        "preprocess_text_embeds 设备和 dtype 同步"
    )
