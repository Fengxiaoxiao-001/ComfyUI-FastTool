# coding=utf-8
# AnimaBaker.py

import copy
import gc
import hashlib
import importlib.util
import os
import re
import sys

from typing import Dict, Type

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

from .utils import (
    _dtype_name, _device_from_string, _set_runtime_dtype_metadata, _module_floating_dtype_counts, _module_dtype,
    _format_dtype_counts, _dtype_from_string,
    _find_model_path, _find_diffusion_model_from_patcher, _resolve_model_runtime_dtype, _get_clip_patcher,
    _patcher_has_weight_patches, _module_needs_dtype_normalization,
    _find_x_embedder_parameter, _module_device, _patch_anima_runtime_inputs, _force_anima_llm_adapter_device,
    _patch_anima_preprocess_text_embeds
)




class AnimaMutationRegistry:
    

    API_VERSION = 1

    REQUIRED_ATTRIBUTES = (
        "MUTATION_API_VERSION",
        "MUTATION_ID",
        "DISPLAY_NAME",
        "MODULE_NAMESPACE",
    )

    REQUIRED_METHODS = (
        "detect",
        "is_mutation_key",
        "install",
    )

    FILENAME_PATTERN = re.compile(
        r"^[a-z][a-z0-9_]*_v[0-9]+\.py$"
    )

    def __init__(
            self,
            mutation_directory,
    ):
        self.mutation_directory = (
            os.path.abspath(
                mutation_directory
            )
        )

        self.mutations: Dict[
            str,
            Type,
        ] = {}

        self.modules: Dict[
            str,
            object,
        ] = {}

    def scan(self):
        self.mutations = {}
        self.modules = {}

        if not os.path.isdir(
                self.mutation_directory
        ):
            os.makedirs(
                self.mutation_directory,
                exist_ok=True,
            )

            print(
                "ℹ️ [AnimaBaker] Mutation 目录不存在，"
                f"已创建: {self.mutation_directory}"
            )

            return self.mutations

        filenames = sorted(
            os.listdir(
                self.mutation_directory
            )
        )

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            if filename.startswith("_"):
                continue

            if not self.FILENAME_PATTERN.match(
                    filename
            ):
                print(
                    "⚠️ [AnimaBaker] Mutation 文件名"
                    "不符合规范，跳过："
                    f"{filename}\n"
                    "    正确示例："
                    "anima_spatial_graft_v1.py"
                )
                continue

            full_path = os.path.join(
                self.mutation_directory,
                filename,
            )

            try:
                mutation_class, module = (
                    self._load_mutation_file(
                        full_path
                    )
                )
            except Exception as exception:
                print(
                    "⚠️ [AnimaBaker] 无法加载 Mutation 文件: "
                    f"{filename}\n"
                    f"    原因: {exception}"
                )
                continue

            mutation_id = str(
                mutation_class.MUTATION_ID
            ).strip()

            filename_id = os.path.splitext(
                filename
            )[0]

            if mutation_id != filename_id:
                raise RuntimeError(
                    "Mutation 的 MUTATION_ID "
                    "必须与文件名一致：\n"
                    f"  文件名: {filename_id}\n"
                    f"  MUTATION_ID: {mutation_id}"
                )

            if mutation_id in self.mutations:
                previous = self.mutations[
                    mutation_id
                ]

                raise RuntimeError(
                    "发现重复的 MUTATION_ID："
                    f"{mutation_id}\n"
                    f"已存在类: {previous}\n"
                    f"重复文件: {full_path}"
                )

            self.mutations[
                mutation_id
            ] = mutation_class

            self.modules[
                mutation_id
            ] = module

            display_name = getattr(
                mutation_class,
                "DISPLAY_NAME",
                mutation_id,
            )

            print(
                "✅ [AnimaBaker] 已注册 Mutation: "
                f"{display_name} "
                f"[{mutation_id}] "
                f"<- {filename}"
            )

        return self.mutations

    def _load_mutation_file(
            self,
            path,
    ):
        basename = os.path.basename(
            path
        )

        module_hash = hashlib.sha1(
            os.path.abspath(path).encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        module_name = (
            f"anima_mutation_{module_hash}"
        )

        spec = (
            importlib.util
            .spec_from_file_location(
                module_name,
                path,
            )
        )

        if (
                spec is None
                or spec.loader is None
        ):
            raise ImportError(
                "无法为 Mutation 创建模块规范: "
                f"{path}"
            )

        module = (
            importlib.util
            .module_from_spec(spec)
        )


        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(
                module
            )
        except Exception:
            sys.modules.pop(
                module_name,
                None,
            )
            raise

        mutation_class = getattr(
            module,
            "GraftedAnima",
            None,
        )

        if mutation_class is None:
            raise AttributeError(
                f"{basename} 中不存在 "
                "GraftedAnima 类"
            )

        for attribute_name in (
                self.REQUIRED_ATTRIBUTES
        ):
            if not hasattr(
                    mutation_class,
                    attribute_name,
            ):
                raise TypeError(
                    f"{basename} 的 GraftedAnima "
                    f"缺少属性: {attribute_name}"
                )

        api_version = getattr(
            mutation_class,
            "MUTATION_API_VERSION",
            None,
        )

        if api_version != self.API_VERSION:
            raise RuntimeError(
                f"{basename} 使用了不兼容的 "
                "MUTATION_API_VERSION："
                f"{api_version}，"
                f"当前要求 {self.API_VERSION}"
            )

        mutation_id = getattr(
            mutation_class,
            "MUTATION_ID",
            None,
        )

        if not isinstance(
                mutation_id,
                str,
        ):
            raise TypeError(
                f"{basename} 的 GraftedAnima "
                "必须声明字符串 MUTATION_ID"
            )

        namespace = getattr(
            mutation_class,
            "MODULE_NAMESPACE",
            None,
        )

        if (
                not isinstance(namespace, str)
                or not namespace.strip()
        ):
            raise TypeError(
                f"{basename} 的 GraftedAnima "
                "必须声明非空 MODULE_NAMESPACE"
            )

        for method_name in self.REQUIRED_METHODS:
            method = getattr(
                mutation_class,
                method_name,
                None,
            )

            if (
                    method is None
                    or not callable(method)
            ):
                raise TypeError(
                    f"{basename} 的 GraftedAnima "
                    f"缺少可调用方法: {method_name}"
                )

        return mutation_class, module

    def detect_for_keys(
            self,
            keys,
            source_name,
    ):
        keys = list(keys)
        matches = []

        for (
                mutation_id,
                mutation_class,
        ) in self.mutations.items():
            try:
                score = (
                    mutation_class.detect(
                        keys
                    )
                )
            except Exception as exception:
                print(
                    "⚠️ [AnimaBaker] Mutation 检测失败: "
                    f"{mutation_id}, "
                    f"source={source_name}, "
                    f"error={exception}"
                )
                continue

            if isinstance(score, bool):
                score = (
                    100
                    if score
                    else 0
                )

            try:
                score = int(score)
            except Exception:
                score = 0

            if score > 0:
                matches.append(
                    (
                        score,
                        mutation_id,
                        mutation_class,
                    )
                )

        if not matches:
            return None

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score = matches[0][0]

        best_matches = [
            item
            for item in matches
            if item[0] == best_score
        ]

        if len(best_matches) > 1:
            names = [
                item[1]
                for item in best_matches
            ]

            raise RuntimeError(
                "无法唯一判断 Anima Mutation。\n"
                f"来源: {source_name}\n"
                f"候选: {names}\n"
                f"得分: {best_score}\n"
                "请增强各 Mutation 文件中的 detect()。"
            )

        (
            score,
            mutation_id,
            mutation_class,
        ) = best_matches[0]

        print(
            "🧬 [AnimaBaker] 检测到 Mutation: "
            f"{mutation_id} "
            f"| source={source_name} "
            f"| score={score}"
        )

        return mutation_class

    @staticmethod
    def resolve_detected_mutations(
            detected_sources,
    ):
        mutation_sources = {}

        for (
                source_name,
                mutation_class,
        ) in detected_sources:
            if mutation_class is None:
                continue

            mutation_id = (
                mutation_class.MUTATION_ID
            )

            mutation_sources.setdefault(
                mutation_id,
                {
                    "class": mutation_class,
                    "sources": [],
                },
            )

            mutation_sources[
                mutation_id
            ]["sources"].append(
                source_name
            )

        if not mutation_sources:
            return None

        if len(mutation_sources) > 1:
            descriptions = []

            for (
                    mutation_id,
                    payload,
            ) in mutation_sources.items():
                descriptions.append(
                    f"  - {mutation_id}: "
                    f"{payload['sources']}"
                )

            raise RuntimeError(
                "当前模型和 LoRA 同时包含多个不兼容的 "
                "Anima Mutation：\n"
                + "\n".join(descriptions)
                + "\n一次烧录只能使用一种主 Mutation。"
            )

        return next(
            iter(
                mutation_sources.values()
            )
        )["class"]


def _extract_mutation_tensors(
        state_dict,
        mutation_classes,
):
    result = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        for mutation_class in mutation_classes:
            try:
                is_mutation = (
                    mutation_class
                    .is_mutation_key(key)
                )
            except Exception:
                is_mutation = False

            if is_mutation:
                result[key] = value
                break

    return result


def _normalize_state_key(
        source_key,
):
    normalized = str(source_key)

    prefixes = (
        "module.",
        "state_dict.",
        "base_model.model.",
        "base_model.",
        "model.model.diffusion_model.",
        "model.diffusion_model.",
        "model.model.",
        "diffusion_model.",
        "model.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if normalized.startswith(
                    prefix
            ):
                normalized = normalized[
                    len(prefix):
                ]

                changed = True
                break

    return normalized


def _find_target_key_by_suffix(
        source_key,
        target_keys,
):
    source_key = str(source_key)

    if source_key in target_keys:
        return source_key

    normalized = _normalize_state_key(
        source_key
    )

    if normalized in target_keys:
        return normalized

    matches = []

    for target_key in target_keys:
        if (
                source_key.endswith(
                    "." + target_key
                )
                or normalized.endswith(
            "." + target_key
        )
        ):
            matches.append(
                target_key
            )

    if len(matches) == 1:
        return matches[0]

    return None


def _validate_tensor_finite(
        tensor,
        key,
):
    if not torch.is_tensor(tensor):
        return

    if not tensor.is_floating_point():
        return

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            "参数中检测到 NaN/Inf：\n"
            f"  {key}\n"
            "该参数可能直接导致纯黑图。"
        )


def _load_mutation_base_weights(
        diffusion_model,
        mutation_class,
        source_state_dict,
        require_complete=False,
):
    if not source_state_dict:
        if require_complete:
            raise RuntimeError(
                "基础模型被识别为 Mutation 模型，"
                "但没有保存任何 Mutation 参数。"
            )

        return 0

    target_state_dict = (
        diffusion_model.state_dict()
    )

    target_keys = set(
        target_state_dict.keys()
    )

    target_mutation_keys = {
        key
        for key in target_keys
        if mutation_class.is_mutation_key(
            key
        )
    }

    load_state_dict = {}
    shape_mismatches = []
    unmapped_keys = []

    for (
            source_key,
            source_tensor,
    ) in source_state_dict.items():
        try:
            if not (
                    mutation_class
                            .is_mutation_key(
                        source_key
                    )
            ):
                continue
        except Exception:
            continue

        _validate_tensor_finite(
            source_tensor,
            source_key,
        )

        target_key = (
            _find_target_key_by_suffix(
                source_key,
                target_keys,
            )
        )

        if target_key is None:
            unmapped_keys.append(
                source_key
            )
            continue

        target_tensor = (
            target_state_dict[
                target_key
            ]
        )

        if (
                tuple(source_tensor.shape)
                != tuple(target_tensor.shape)
        ):
            shape_mismatches.append(
                (
                    source_key,
                    tuple(
                        source_tensor.shape
                    ),
                    target_key,
                    tuple(
                        target_tensor.shape
                    ),
                )
            )
            continue

        load_state_dict[
            target_key
        ] = source_tensor

    if shape_mismatches:
        preview = "\n".join(
            (
                f"  - {source_key} "
                f"{source_shape} -> "
                f"{target_key} "
                f"{target_shape}"
            )
            for (
                source_key,
                source_shape,
                target_key,
                target_shape,
            ) in shape_mismatches[:30]
        )

        raise RuntimeError(
            "Mutation 参数形状不匹配：\n"
            + preview
        )

    missing_target_keys = sorted(
        target_mutation_keys
        - set(load_state_dict.keys())
    )

    if require_complete and unmapped_keys:
        preview = "\n".join(
            f"  - {key}"
            for key in unmapped_keys[:30]
        )

        raise RuntimeError(
            "基础 Mutation checkpoint 中存在"
            "无法映射的变体参数。\n"
            "为防止部分随机初始化导致黑图，已停止加载。\n"
            "无法映射的键示例：\n"
            f"{preview}"
        )

    if (
            require_complete
            and missing_target_keys
    ):
        preview = "\n".join(
            f"  - {key}"
            for key in missing_target_keys[:30]
        )

        raise RuntimeError(
            "安装后的 Mutation 架构缺少 checkpoint "
            "对应权重。\n"
            "为防止随机初始化层导致黑图，已停止加载。\n"
            "缺失目标键示例：\n"
            f"{preview}"
        )

    if (
            require_complete
            and not load_state_dict
    ):
        raise RuntimeError(
            "基础模型包含 Mutation 参数，"
            "但安装架构后没有任何参数能够映射。"
        )

    incompatible = (
        diffusion_model.load_state_dict(
            load_state_dict,
            strict=False,
        )
    )

    unexpected = list(
        incompatible.unexpected_keys
    )

    if unexpected:
        raise RuntimeError(
            "Mutation 二次加载仍出现 unexpected keys：\n"
            + "\n".join(
                f"  - {key}"
                for key in unexpected[:30]
            )
        )

    if (
            unmapped_keys
            and not require_complete
    ):
        print(
            "⚠️ [AnimaBaker] 有 "
            f"{len(unmapped_keys)} 个 Mutation 参数"
            "无法映射。"
        )

    print(
        "✅ [AnimaBaker] 已完整加载基础模型 "
        "Mutation 参数: "
        f"{len(load_state_dict)}"
    )

    return len(load_state_dict)


def _count_mutation_patches(
        model_patcher,
        mutation_class,
):
    patches = getattr(
        model_patcher,
        "patches",
        {},
    )

    count = 0

    for key in patches.keys():
        try:
            if (
                    mutation_class
                            .is_mutation_key(
                        str(key)
                    )
            ):
                count += 1
        except Exception:
            pass

    return count




@torch.inference_mode()
def _bake_patcher_weights(
        patcher,
        bake_dtype,
        calc_device,
        name,
):


    if patcher is None:
        return patcher

    if bake_dtype is None:
        raise ValueError(
            f"{name} bake_dtype 不能为 None"
        )

    if bake_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
    ):
        raise ValueError(
            f"{name} 不支持的 bake_dtype："
            f"{bake_dtype}"
        )

    print(
        "🔥 [AnimaBaker] 使用 "
        f"[{calc_device}] "
        f"以 {_dtype_name(bake_dtype)} "
        f"低显存烧录 {name}..."
    )

    model_inner = patcher.model


    model_inner.to(
        device=torch.device("cpu")
    )

    gc.collect()

    (
        comfy.model_management
        .soft_empty_cache()
    )

    state_dict = (
        model_inner.state_dict()
    )

    patches = getattr(
        patcher,
        "patches",
        {},
    )

    state_keys = set(
        state_dict.keys()
    )

    unused_patch_keys = sorted(
        set(patches.keys())
        - state_keys
    )

    if unused_patch_keys:
        preview = "\n".join(
            f"  - {key}"
            for key in unused_patch_keys[:30]
        )

        raise RuntimeError(
            f"{name} 中存在无法对应模型参数的 LoRA patch：\n"
            f"{preview}\n"
            "这通常说明 LoRA 键映射错误，继续烧录可能"
            "产生不完整模型。"
        )

    new_state_dict = {}

    total = len(state_dict)

    for index, (
            key,
            source_tensor,
    ) in enumerate(
        state_dict.items()
    ):
        if index % 100 == 0:
            print(
                f"  - {name}: "
                f"{index + 1}/{total}"
            )

        if not torch.is_tensor(
                source_tensor
        ):
            continue

        if source_tensor.is_floating_point():

            weight = source_tensor.detach().to(
                device=calc_device,
                dtype=torch.float32,
            )

            if key in patches:
                weight = (
                    comfy.lora
                    .calculate_weight(
                        patches[key],
                        weight,
                        key,
                    )
                )

            if not torch.is_tensor(weight):
                raise RuntimeError(
                    f"烧录 {name} 时 "
                    f"calculate_weight({key}) "
                    "没有返回 Tensor"
                )

            if not torch.isfinite(
                    weight
            ).all():
                raise RuntimeError(
                    f"烧录 {name} 时参数出现 NaN/Inf：\n"
                    f"  {key}\n"
                    "已停止烧录，避免生成纯黑图。"
                )

            if bake_dtype == torch.float16:
                fp16_limit = (
                    torch.finfo(
                        torch.float16
                    ).max
                )

                max_value = (
                    weight.abs().max().item()
                    if weight.numel() > 0
                    else 0.0
                )

                if max_value > fp16_limit:
                    raise RuntimeError(
                        "参数无法安全转换为 float16：\n"
                        f"  key={key}\n"
                        f"  abs_max={max_value}\n"
                        f"  float16_max={fp16_limit}\n"
                        "请把 save_dtype 改为 "
                        "bfloat16 或 float32。"
                    )

            converted = weight.to(
                dtype=bake_dtype
            )

            if not torch.isfinite(
                    converted
            ).all():
                raise RuntimeError(
                    f"参数转换为 "
                    f"{_dtype_name(bake_dtype)} "
                    "后出现 NaN/Inf：\n"
                    f"  {key}"
                )

            new_state_dict[key] = (
                converted.detach()
                .to(device="cpu")
                .contiguous()
            )

            del converted
            del weight
        else:
            if key in patches:
                raise RuntimeError(
                    "LoRA patch 错误地指向"
                    "非浮点参数：\n"
                    f"  {key}"
                )

            new_state_dict[key] = (
                source_tensor.detach()
                .to(device="cpu")
                .clone()
            )


        if (
                index > 0
                and index % 200 == 0
        ):
            (
                comfy.model_management
                .soft_empty_cache()
            )


    model_inner.to(
        device=torch.device("cpu"),
        dtype=bake_dtype,
    )

    incompatible = (
        model_inner.load_state_dict(
            new_state_dict,
            strict=False,
        )
    )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"{name} 烧录后出现 unexpected keys：\n"
            + "\n".join(
                f"  - {key}"
                for key in (
                    incompatible
                    .unexpected_keys[:30]
                )
            )
        )


    if incompatible.missing_keys:
        preview = "\n".join(
            f"  - {key}"
            for key in (
                incompatible
                .missing_keys[:30]
            )
        )

        raise RuntimeError(
            f"{name} 烧录后出现 missing keys：\n"
            f"{preview}"
        )


    _set_runtime_dtype_metadata(
        patcher,
        model_inner,
        bake_dtype,
    )


    patcher.patches = {}
    patcher.backup = {}

    del new_state_dict
    del state_dict

    gc.collect()

    (
        comfy.model_management
        .soft_empty_cache()
    )

    final_counts = (
        _module_floating_dtype_counts(
            model_inner
        )
    )

    print(
        f"✅ [AnimaBaker] {name} 烧录完成，"
        "模型保留在 CPU；最终 Parameter dtype："
        f"{_format_dtype_counts(final_counts)}"
    )

    return patcher




class _AnimaBakerCore:
    def _run_baker(
            self,
            model,
            clip,
            vae,
            lora_stack=None,
            save_dtype="auto",
            device="auto",
            enable_mutation=False,
    ):
        lora_stack = (
            []
            if lora_stack is None
            else lora_stack
        )

        active_loras = [
            lora
            for lora in lora_stack
            if (
                    lora[0]
                    and lora[0] != "None"
                    and (
                            abs(float(lora[1]))
                            > 0.0001
                            or
                            abs(float(lora[2]))
                            > 0.0001
                    )
            )
        ]

        print(
            "🧹 [AnimaBaker] 清空残留显存..."
        )

        (
            comfy.model_management
            .unload_all_models()
        )

        gc.collect()

        (
            comfy.model_management
            .soft_empty_cache()
        )

        target_device = (
            _device_from_string(
                device
            )
        )

        print(
            "🔧 [AnimaBaker] 使用设备: "
            f"{target_device}"
        )

        model_path, _ = _find_model_path(
            model,
            ["checkpoints"],
        )

        if model_path is None:
            raise RuntimeError(
                f"找不到主模型文件: {model}"
            )

        vae_path, _ = _find_model_path(
            vae,
            ["vae"],
        )

        if vae_path is None:
            raise RuntimeError(
                f"找不到 VAE 模型文件: {vae}"
            )

        clip_path, _ = _find_model_path(
            clip,
            ["clip"],
        )

        if clip_path is None:
            raise RuntimeError(
                f"找不到 CLIP 模型文件: {clip}"
            )



        selected_mutation_class = None
        checkpoint_mutation_class = None

        mutation_base_tensors = {}
        lora_mutation_map = {}
        lora_key_cache = {}

        if enable_mutation:
            mutation_directory = os.path.join(
                os.path.dirname(
                    os.path.abspath(
                        __file__
                    )
                ),
                "Mutation",
            )

            mutation_registry = (
                AnimaMutationRegistry(
                    mutation_directory
                )
            )

            mutation_registry.scan()

            if mutation_registry.mutations:
                print(
                    "🔍 [AnimaBaker] 扫描基础模型 "
                    "Mutation 参数..."
                )

                checkpoint_scan_sd = (
                    comfy.utils
                    .load_torch_file(
                        model_path,
                        safe_load=True,
                    )
                )

                checkpoint_mutation_class = (
                    mutation_registry
                    .detect_for_keys(
                        checkpoint_scan_sd.keys(),
                        source_name=(
                            f"checkpoint:{model}"
                        ),
                    )
                )

                mutation_base_tensors = (
                    _extract_mutation_tensors(
                        checkpoint_scan_sd,
                        list(
                            mutation_registry
                            .mutations
                            .values()
                        ),
                    )
                )

                del checkpoint_scan_sd
                gc.collect()

                detected_sources = [
                    (
                        f"checkpoint:{model}",
                        checkpoint_mutation_class,
                    )
                ]

                for (
                        lora_name,
                        model_strength,
                        clip_strength,
                ) in active_loras:
                    lora_path, _ = (
                        _find_model_path(
                            lora_name,
                            ["loras"],
                        )
                    )

                    if lora_path is None:
                        raise RuntimeError(
                            "找不到 LoRA："
                            f"{lora_name}"
                        )

                    print(
                        "🔍 [AnimaBaker] 扫描 LoRA 架构: "
                        f"{lora_name}"
                    )

                    lora_scan_sd = (
                        comfy.utils
                        .load_torch_file(
                            lora_path,
                            safe_load=True,
                        )
                    )


                    lora_keys = list(
                        lora_scan_sd.keys()
                    )

                    lora_key_cache[
                        lora_name
                    ] = lora_keys

                    lora_mutation_class = (
                        mutation_registry
                        .detect_for_keys(
                            lora_keys,
                            source_name=(
                                f"lora:{lora_name}"
                            ),
                        )
                    )

                    lora_mutation_map[
                        lora_name
                    ] = lora_mutation_class

                    detected_sources.append(
                        (
                            f"lora:{lora_name}",
                            lora_mutation_class,
                        )
                    )

                    del lora_scan_sd
                    gc.collect()

                selected_mutation_class = (
                    mutation_registry
                    .resolve_detected_mutations(
                        detected_sources
                    )
                )

                if (
                        selected_mutation_class
                        is None
                ):
                    print(
                        "ℹ️ [AnimaBaker] 未检测到 "
                        "Mutation，按原版 Anima 处理"
                    )
                else:
                    print(
                        "🧬 [AnimaBaker] 最终使用 Mutation: "
                        f"{selected_mutation_class.MUTATION_ID}"
                    )
            else:
                print(
                    "ℹ️ [AnimaBaker] Mutation 目录中"
                    "没有可用变体"
                )



        print(
            "📥 [AnimaBaker] 加载 Anima 基础模型: "
            f"{model_path}"
        )

        if checkpoint_mutation_class is not None:
            print(
                "ℹ️ [AnimaBaker] 接下来 ComfyUI 可能报告 "
                "spatial_graft unexpected keys。\n"
                "这是首次按原版 Anima 建模时的预期提示；"
                "Mutation 参数随后会被严格二次加载。"
            )

        ckpt_out = (
            comfy.sd
            .load_checkpoint_guess_config(
                model_path,
                output_vae=False,
                output_clip=False,
            )
        )

        model_obj = ckpt_out[0]

        print(
            "📥 [AnimaBaker] 加载 Anima VAE: "
            f"{vae_path}"
        )

        vae_sd = (
            comfy.utils.load_torch_file(
                vae_path
            )
        )

        vae_obj = comfy.sd.VAE(
            sd=vae_sd
        )

        del vae_sd
        gc.collect()



        print(
            "📥 [AnimaBaker] 加载 Anima Qwen3 CLIP: "
            f"{clip_path}"
        )

        if (
                AnimaTEModel is None
                or AnimaTokenizer is None
        ):
            raise RuntimeError(
                "当前 ComfyUI 环境中找不到 "
                "comfy.text_encoders.anima."
                "AnimaTEModel / AnimaTokenizer"
            )

        clip_sd = (
            comfy.utils.load_torch_file(
                clip_path,
                safe_load=True,
            )
        )

        clip_target = (
            comfy.supported_models_base
            .ClipTarget(
                tokenizer=AnimaTokenizer,
                clip=AnimaTEModel,
            )
        )

        clip_obj = CLIP(
            clip_target,
            embedding_directory=None,
        )

        clip_obj.load_sd(
            clip_sd
        )

        del clip_sd
        gc.collect()

        model_clone = model_obj.clone()

        if hasattr(
                clip_obj,
                "clone",
        ):
            clip_clone = clip_obj.clone()
        else:
            clip_clone = copy.copy(
                clip_obj
            )

        diffusion_model = (
            _find_diffusion_model_from_patcher(
                model_clone
            )
        )

        model_parameter_dtype = (
            _module_dtype(
                diffusion_model,
                torch.bfloat16,
            )
        )


        model_base_dtype = (
            _resolve_model_runtime_dtype(
                model_clone,
                diffusion_model,
                model_parameter_dtype,
            )
        )

        model_target_dtype = (
            _dtype_from_string(
                save_dtype,
                model_base_dtype,
            )
        )

        if model_target_dtype is None:
            model_target_dtype = (
                model_base_dtype
            )

        clip_patcher = (
            _get_clip_patcher(
                clip_clone
            )
        )

        clip_model_inner = (
            clip_patcher.model
            if clip_patcher is not None
            else None
        )

        clip_parameter_dtype = (
            _module_dtype(
                clip_model_inner,
                torch.bfloat16,
            )
        )

        clip_base_dtype = (
            _resolve_model_runtime_dtype(
                clip_patcher,
                clip_model_inner,
                clip_parameter_dtype,
            )
        )


        clip_target_dtype = (
            _dtype_from_string(
                save_dtype,
                clip_base_dtype,
            )
        )

        if clip_target_dtype is None:
            clip_target_dtype = (
                clip_base_dtype
            )

        model_dtype_counts = (
            _module_floating_dtype_counts(
                model_clone.model
            )
        )

        clip_dtype_counts = (
            _module_floating_dtype_counts(
                clip_model_inner
            )
        )

        print(
            "🔧 [AnimaBaker] 主模型运行 dtype: "
            f"{_dtype_name(model_base_dtype)} "
            "-> "
            f"{_dtype_name(model_target_dtype)}"
        )

        print(
            "🔎 [AnimaBaker] 主模型 Parameter "
            "dtype 分布: "
            f"{_format_dtype_counts(model_dtype_counts)}"
        )

        print(
            "🔧 [AnimaBaker] CLIP 运行 dtype: "
            f"{_dtype_name(clip_base_dtype)} "
            "-> "
            f"{_dtype_name(clip_target_dtype)}"
        )

        print(
            "🔎 [AnimaBaker] CLIP Parameter "
            "dtype 分布: "
            f"{_format_dtype_counts(clip_dtype_counts)}"
        )



        if selected_mutation_class is not None:
            print(
                "🧬 [AnimaBaker] 正在安装 Mutation 架构: "
                f"{selected_mutation_class.MUTATION_ID}"
            )

            source_keys = list(
                mutation_base_tensors.keys()
            )

            for lora_name, lora_keys in (
                    lora_key_cache.items()
            ):
                if (
                        lora_mutation_map.get(
                            lora_name
                        )
                        is not None
                ):
                    source_keys.extend(
                        lora_keys
                    )

            selected_mutation_class.install(
                diffusion_model,
                source_keys=source_keys,
                source_state_dict=(
                    mutation_base_tensors
                ),
            )

            installed_id = getattr(
                diffusion_model,
                "_anima_mutation_id",
                None,
            )

            if (
                    installed_id
                    != selected_mutation_class
                    .MUTATION_ID
            ):
                raise RuntimeError(
                    "Mutation install() 执行结束后，"
                    "没有正确设置模型标记。\n"
                    "期望: "
                    f"{selected_mutation_class.MUTATION_ID}\n"
                    f"实际: {installed_id}"
                )

            _load_mutation_base_weights(
                diffusion_model,
                selected_mutation_class,
                mutation_base_tensors,
                require_complete=(
                        checkpoint_mutation_class
                        is not None
                ),
            )

        del mutation_base_tensors
        gc.collect()



        print(
            "🛠️ [AnimaBaker] LoRA 数量: "
            f"{len(active_loras)}"
        )

        for (
                lora_name,
                model_strength,
                clip_strength,
        ) in active_loras:
            lora_path, _ = (
                _find_model_path(
                    lora_name,
                    ["loras"],
                )
            )

            if lora_path is None:
                raise RuntimeError(
                    "找不到 LoRA："
                    f"{lora_name}"
                )

            print(
                "  - 应用 LoRA: "
                f"{lora_name} "
                f"| model={model_strength}, "
                f"clip={clip_strength}"
            )

            lora_sd = (
                comfy.utils
                .load_torch_file(
                    lora_path,
                    safe_load=True,
                )
            )

            mutation_patch_count_before = 0

            if (
                    selected_mutation_class
                    is not None
            ):
                mutation_patch_count_before = (
                    _count_mutation_patches(
                        model_clone,
                        selected_mutation_class,
                    )
                )

            if hasattr(
                    clip_clone,
                    "cond_stage_model",
            ):
                (
                    model_clone,
                    clip_clone,
                ) = (
                    comfy.sd
                    .load_lora_for_models(
                        model_clone,
                        clip_clone,
                        lora_sd,
                        float(model_strength),
                        float(clip_strength),
                    )
                )
            else:
                model_clone, _ = (
                    comfy.sd
                    .load_lora_for_models(
                        model_clone,
                        None,
                        lora_sd,
                        float(model_strength),
                        float(clip_strength),
                    )
                )

            if (
                    selected_mutation_class
                    is not None
                    and lora_mutation_map.get(
                lora_name
            ) is not None
                    and abs(
                float(model_strength)
            ) > 0.0001
            ):
                mutation_patch_count_after = (
                    _count_mutation_patches(
                        model_clone,
                        selected_mutation_class,
                    )
                )

                added_count = (
                        mutation_patch_count_after
                        - mutation_patch_count_before
                )

                if added_count <= 0:
                    mutation_keys = [
                        key
                        for key in lora_sd.keys()
                        if (
                            selected_mutation_class
                            .is_mutation_key(key)
                        )
                    ]

                    preview = "\n".join(
                        f"    - {key}"
                        for key in (
                            mutation_keys[:20]
                        )
                    )

                    raise RuntimeError(
                        f"LoRA {lora_name} 被识别为 "
                        f"{selected_mutation_class.MUTATION_ID}，"
                        "但 ComfyUI 没有为新增层生成任何 patch。\n"
                        "相关键示例：\n"
                        f"{preview}\n"
                        "请确认该 LoRA 使用真实模块路径保存，"
                        "并且键名能被 ComfyUI LoRA 映射器识别。"
                    )

                print(
                    "    ✅ Mutation LoRA patch "
                    f"数量增加: {added_count}"
                )

            del lora_sd
            gc.collect()


        clip_patcher = (
            _get_clip_patcher(
                clip_clone
            )
        )



        model_has_patches = (
            _patcher_has_weight_patches(
                model_clone
            )
        )

        clip_has_patches = (
            _patcher_has_weight_patches(
                clip_patcher
            )
        )

        model_dtype_mismatch = (
            _module_needs_dtype_normalization(
                model_clone.model,
                model_target_dtype,
            )
        )

        clip_dtype_mismatch = (
            _module_needs_dtype_normalization(
                (
                    clip_patcher.model
                    if clip_patcher is not None
                    else None
                ),
                clip_target_dtype,
            )
        )

        model_needs_bake = (
                model_has_patches
                or model_target_dtype
                != model_base_dtype
                or model_dtype_mismatch
        )

        clip_needs_bake = (
                clip_has_patches
                or clip_target_dtype
                != clip_base_dtype
                or clip_dtype_mismatch
        )

        if model_dtype_mismatch:
            print(
                "⚠️ [AnimaBaker] 检测到主模型内部存在"
                "混合 Parameter dtype，将统一为 "
                f"{_dtype_name(model_target_dtype)}"
            )

        if clip_dtype_mismatch:
            print(
                "⚠️ [AnimaBaker] 检测到 CLIP 内部存在"
                "混合 Parameter dtype，将统一为 "
                f"{_dtype_name(clip_target_dtype)}"
            )

        if model_needs_bake:
            model_clone = (
                _bake_patcher_weights(
                    model_clone,
                    bake_dtype=(
                        model_target_dtype
                    ),
                    calc_device=(
                        target_device
                    ),
                    name="UNET/Transformer",
                )
            )
        else:
            print(
                "ℹ️ [AnimaBaker] 主模型没有待烧录 patch，"
                "且 dtype 不变，跳过主模型重写"
            )

        if clip_needs_bake:
            if clip_patcher is None:
                raise RuntimeError(
                    "CLIP 需要烧录，但不存在 patcher"
                )

            clip_clone.patcher = (
                _bake_patcher_weights(
                    clip_patcher,
                    bake_dtype=(
                        clip_target_dtype
                    ),
                    calc_device=(
                        target_device
                    ),
                    name="CLIP",
                )
            )
        else:
            print(
                "ℹ️ [AnimaBaker] CLIP 没有待烧录 patch，"
                "且 dtype 不变，保留原始 CLIP 精度"
            )



        diffusion_model = (
            _find_diffusion_model_from_patcher(
                model_clone
            )
        )

        final_model_dtype_counts = (
            _module_floating_dtype_counts(
                model_clone.model
            )
        )

        if _module_needs_dtype_normalization(
                model_clone.model,
                model_target_dtype,
        ):
            raise RuntimeError(
                "主模型烧录后仍存在混合 Parameter dtype："
                f"{_format_dtype_counts(final_model_dtype_counts)}\n"
                "目标 dtype："
                f"{_dtype_name(model_target_dtype)}"
            )


        _set_runtime_dtype_metadata(
            model_clone,
            diffusion_model,
            model_target_dtype,
        )

        x_embedder_parameter = (
            _find_x_embedder_parameter(
                diffusion_model
            )
        )

        if x_embedder_parameter is None:
            diffusion_dtype = (
                _module_dtype(
                    diffusion_model,
                    model_target_dtype,
                )
            )

            current_model_device = (
                _module_device(
                    diffusion_model,
                    torch.device("cpu"),
                )
            )
        else:
            diffusion_dtype = (
                x_embedder_parameter.dtype
            )

            current_model_device = (
                x_embedder_parameter.device
            )

        if diffusion_dtype != model_target_dtype:
            raise RuntimeError(
                "x_embedder 的实际 dtype 与目标 dtype "
                "不一致：\n"
                f"  x_embedder={_dtype_name(diffusion_dtype)}\n"
                f"  target={_dtype_name(model_target_dtype)}"
            )

        print(
            "✅ [AnimaBaker] 最终主模型运行 dtype: "
            f"{_dtype_name(diffusion_dtype)}"
        )

        print(
            "✅ [AnimaBaker] 最终主模型 Parameter "
            "dtype 分布: "
            f"{_format_dtype_counts(final_model_dtype_counts)}"
        )


        _force_anima_llm_adapter_device(
            diffusion_model,
            current_model_device,
            diffusion_dtype,
        )

        _patch_anima_preprocess_text_embeds(
            diffusion_model
        )


        _patch_anima_runtime_inputs(
            diffusion_model
        )

        gc.collect()

        (
            comfy.model_management
            .soft_empty_cache()
        )

        if selected_mutation_class is None:
            print(
                "✅ [AnimaBaker] 原版 Anima 处理完成"
            )
        else:
            print(
                "✅ [AnimaBaker] Mutation Anima 处理完成: "
                f"{selected_mutation_class.MUTATION_ID}"
            )

        return (
            model_clone,
            clip_clone,
            vae_obj,
        )




class SeparateModelMixerDictFuser(
    _AnimaBakerCore
):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    folder_paths
                    .get_filename_list(
                        "checkpoints"
                    ),
                ),
                "clip": (
                    folder_paths
                    .get_filename_list(
                        "clip"
                    ),
                ),
                "vae": (
                    folder_paths
                    .get_filename_list(
                        "vae"
                    ),
                ),
            },
            "optional": {
                "lora_stack": (
                    "LORA_STACK",
                ),
                "save_dtype": (
                    [
                        "auto",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    {
                        "default": "auto",
                    },
                ),
                "device": (
                    [
                        "auto",
                        "cpu",
                        "cuda",
                        "npu",
                    ],
                    {
                        "default": "auto",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    FUNCTION = "pure_dict_merge"

    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def pure_dict_merge(
            self,
            model,
            clip,
            vae,
            lora_stack=None,
            save_dtype="auto",
            device="auto",
    ):
        return self._run_baker(
            model=model,
            clip=clip,
            vae=vae,
            lora_stack=lora_stack,
            save_dtype=save_dtype,
            device=device,
            enable_mutation=False,
        )




class MutationAnimaModelBaker(
    _AnimaBakerCore
):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    folder_paths
                    .get_filename_list(
                        "checkpoints"
                    ),
                ),
                "clip": (
                    folder_paths
                    .get_filename_list(
                        "clip"
                    ),
                ),
                "vae": (
                    folder_paths
                    .get_filename_list(
                        "vae"
                    ),
                ),
            },
            "optional": {
                "lora_stack": (
                    "LORA_STACK",
                ),
                "save_dtype": (
                    [
                        "auto",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    {
                        "default": "auto",
                    },
                ),
                "device": (
                    [
                        "auto",
                        "cpu",
                        "cuda",
                        "npu",
                    ],
                    {
                        "default": "auto",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    FUNCTION = "mutation_dict_merge"

    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def mutation_dict_merge(
            self,
            model,
            clip,
            vae,
            lora_stack=None,
            save_dtype="auto",
            device="auto",
    ):
        return self._run_baker(
            model=model,
            clip=clip,
            vae=vae,
            lora_stack=lora_stack,
            save_dtype=save_dtype,
            device=device,
            enable_mutation=True,
        )


NODE_CLASS_MAPPINGS = {
    "SeparateModelMixerDictFuser":
        SeparateModelMixerDictFuser,

    "MutationAnimaModelBaker":
        MutationAnimaModelBaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeparateModelMixerDictFuser":
        "Only Anima模型烧录器",

    "MutationAnimaModelBaker":
        "Mutation Anima变体自动烧录器",
}
