# coding=utf-8
# AnimaBaker.py

import copy
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

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
    _dtype_name,
    _device_from_string,
    _set_runtime_dtype_metadata,
    _module_floating_dtype_counts,
    _module_dtype,
    _format_dtype_counts,
    _dtype_from_string,
    _find_model_path,
    _find_diffusion_model_from_patcher,
    _resolve_model_runtime_dtype,
    _get_clip_patcher,
    _patcher_has_weight_patches,
    _module_needs_dtype_normalization,
    _find_x_embedder_parameter,
    _module_device,
    _patch_anima_runtime_inputs,
    _force_anima_llm_adapter_device,
    _patch_anima_preprocess_text_embeds,
)


# ============================================================
# 常量
# ============================================================

ANIMA_GRAFTED4_MUTATION_ID = "anima_spatial_graft_v4"

SUPPORTED_BAKE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

_MODEL_DOTTED_PREFIXES = (
    "module.",
    "state_dict.",
    "base_model.model.",
    "base_model.",
    "model.model.diffusion_model.",
    "model.diffusion_model.",
    "model.model.",
    "diffusion_model.",
    "model.",
    "unet.",
    "transformer.",
    "dit.",
)

_LYCORIS_MODEL_PREFIXES = (
    "lora_unet_",
    "lycoris_unet_",
    "locon_unet_",
    "loha_unet_",
    "lokr_unet_",
    "dylora_unet_",
    "ia3_unet_",
    "oft_unet_",
    "boft_unet_",
    "glora_unet_",

    "lora_transformer_",
    "lycoris_transformer_",
    "locon_transformer_",
    "loha_transformer_",
    "lokr_transformer_",
    "dylora_transformer_",
    "ia3_transformer_",

    "lora_dit_",
    "lycoris_dit_",
    "locon_dit_",
    "loha_dit_",
    "lokr_dit_",

    "unet_",
    "transformer_",
    "dit_",
)

_LYCORIS_MODEL_DOTTED_PREFIXES = (
    "lora_unet.",
    "lycoris_unet.",
    "locon_unet.",
    "loha_unet.",
    "lokr_unet.",
    "dylora_unet.",
    "ia3_unet.",
    "oft_unet.",
    "boft_unet.",
    "glora_unet.",

    "lora_transformer.",
    "lycoris_transformer.",
    "locon_transformer.",
    "loha_transformer.",
    "lokr_transformer.",
    "dylora_transformer.",
    "ia3_transformer.",

    "lora_dit.",
    "lycoris_dit.",
    "locon_dit.",
    "loha_dit.",
    "lokr_dit.",
)

_CURRENT_GRAFT_TOKENS = (
    ".mudd_graft.",
    ".spatial_graft.",
    ".mor_graft.",
    "_mudd_graft_",
    "_spatial_graft_",
    "_mor_graft_",
    "mudd_graft_",
    "spatial_graft_",
    "mor_graft_",
    ".extra_blocks.",
    "extra_blocks.",
    "_extra_blocks_",
    "extra_blocks_",
)


# ============================================================
# 通用工具
# ============================================================

def _safe_gc():
    gc.collect()

    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


def _get_state_dict_from_owner(
        owner: Any,
) -> Mapping[str, torch.Tensor]:
    candidates = (
        owner,
        getattr(owner, "model", None),
    )

    visited: Set[int] = set()

    for candidate in candidates:
        if candidate is None:
            continue

        candidate_id = id(candidate)

        if candidate_id in visited:
            continue

        visited.add(candidate_id)

        state_dict_method = getattr(
            candidate,
            "state_dict",
            None,
        )

        if not callable(state_dict_method):
            continue

        try:
            state_dict = state_dict_method()
        except Exception:
            continue

        if isinstance(state_dict, Mapping):
            return state_dict

    return {}


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
            "该参数可能造成纯黑图或数值异常。"
        )


def _validate_module_parameters_finite(
        module,
        module_name,
):
    if module is None:
        return

    for name, parameter in module.named_parameters(
            recurse=True
    ):
        _validate_tensor_finite(
            parameter,
            f"{module_name}.{name}",
        )


def _normalize_state_key(
        source_key,
):
    normalized = str(source_key).replace("/", ".").strip(".")

    changed = True

    while changed:
        changed = False

        for prefix in _MODEL_DOTTED_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
                break

    return normalized.strip(".")


def _find_target_key_by_suffix(
        source_key,
        target_keys,
):
    source_key = str(source_key)
    normalized_source = _normalize_state_key(
        source_key
    )

    if source_key in target_keys:
        return source_key

    if normalized_source in target_keys:
        return normalized_source

    exact_normalized_matches = []

    for target_key in target_keys:
        normalized_target = _normalize_state_key(
            target_key
        )

        if normalized_source == normalized_target:
            exact_normalized_matches.append(
                target_key
            )

    exact_normalized_matches = list(
        dict.fromkeys(exact_normalized_matches)
    )

    if len(exact_normalized_matches) == 1:
        return exact_normalized_matches[0]

    suffix_matches = []

    for target_key in target_keys:
        normalized_target = _normalize_state_key(
            target_key
        )

        if normalized_source.endswith(
                "." + normalized_target
        ):
            suffix_matches.append(target_key)
            continue

        if normalized_target.endswith(
                "." + normalized_source
        ):
            suffix_matches.append(target_key)

    suffix_matches = list(
        dict.fromkeys(suffix_matches)
    )

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    return None


def _remove_parameter_suffix(
        key: str,
) -> str:
    for suffix in (
            ".weight",
            ".bias",
    ):
        if key.endswith(suffix):
            return key[:-len(suffix)]

    return key


def _contains_current_graft_key(
        key: str,
) -> bool:
    lowered = str(key).replace("/", ".").lower()

    return any(
        token in lowered
        for token in _CURRENT_GRAFT_TOKENS
    )


def _strip_repeated_dotted_prefixes(
        key: str,
) -> str:
    result = str(key).replace("/", ".").strip(".")

    changed = True

    while changed:
        changed = False
        lowered = result.lower()

        for prefix in _MODEL_DOTTED_PREFIXES:
            if lowered.startswith(prefix.lower()):
                result = result[len(prefix):]
                result = result.strip(".")
                changed = True
                break

    return result


def _strip_lycoris_model_prefix(
        key: str,
) -> Tuple[str, bool]:
    result = str(key).replace("/", ".").strip("._")
    lowered = result.lower()

    for prefix in _LYCORIS_MODEL_DOTTED_PREFIXES:
        if lowered.startswith(prefix):
            return (
                result[len(prefix):].strip("._"),
                True,
            )

    for prefix in _LYCORIS_MODEL_PREFIXES:
        if lowered.startswith(prefix):
            return (
                result[len(prefix):].strip("._"),
                True,
            )

    return result, False


def _is_explicit_model_adapter_base(
        source_base: str,
) -> bool:
    value = str(source_base).replace("/", ".").strip()
    lowered = value.lower()

    if lowered.startswith(
            tuple(
                prefix.lower()
                for prefix in _MODEL_DOTTED_PREFIXES
            )
    ):
        return True

    if lowered.startswith(
            _LYCORIS_MODEL_PREFIXES
    ):
        return True

    if lowered.startswith(
            _LYCORIS_MODEL_DOTTED_PREFIXES
    ):
        return True

    if _contains_current_graft_key(lowered):
        return True

    return False


def _mapping_target_to_string(
        value,
) -> Optional[str]:
    if isinstance(value, str):
        return value

    if isinstance(value, Sequence):
        for item in value:
            if isinstance(item, str):
                return item

    return None


# ============================================================
# Mutation 注册系统
# ============================================================

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
        self.mutation_directory = os.path.abspath(
            mutation_directory
        )

        self.mutations: Dict[str, Type] = {}
        self.modules: Dict[str, object] = {}

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
                "ℹ️ [AnimaBaker] 已创建 Mutation 目录: "
                f"{self.mutation_directory}"
            )

            return self.mutations

        for filename in sorted(
                os.listdir(self.mutation_directory)
        ):
            if not filename.endswith(".py"):
                continue

            if filename.startswith("_"):
                continue

            if not self.FILENAME_PATTERN.match(
                    filename
            ):
                print(
                    "⚠️ [AnimaBaker] Mutation 文件名不符合规范，跳过: "
                    f"{filename}"
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
                    "⚠️ [AnimaBaker] Mutation 加载失败: "
                    f"{filename}\n"
                    f"    {exception}"
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
                    "Mutation 的 MUTATION_ID 必须与文件名一致：\n"
                    f"  文件名: {filename_id}\n"
                    f"  MUTATION_ID: {mutation_id}"
                )

            if mutation_id in self.mutations:
                raise RuntimeError(
                    "发现重复的 MUTATION_ID："
                    f"{mutation_id}"
                )

            self.mutations[mutation_id] = (
                mutation_class
            )

            self.modules[mutation_id] = module

            print(
                "✅ [AnimaBaker] 已注册 Mutation: "
                f"{mutation_class.DISPLAY_NAME} "
                f"[{mutation_id}]"
            )

        return self.mutations

    def _load_mutation_file(
            self,
            path,
    ):
        basename = os.path.basename(path)

        module_hash = hashlib.sha1(
            os.path.abspath(path).encode("utf-8")
        ).hexdigest()[:12]

        module_name = (
            f"anima_mutation_{module_hash}"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"无法创建 Mutation 模块规范: {path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )

        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        mutation_class = getattr(
            module,
            "GraftedAnima",
            None,
        )

        if mutation_class is None:
            raise AttributeError(
                f"{basename} 中不存在 GraftedAnima"
            )

        for attribute_name in (
                self.REQUIRED_ATTRIBUTES
        ):
            if not hasattr(
                    mutation_class,
                    attribute_name,
            ):
                raise TypeError(
                    f"{basename} 缺少属性: "
                    f"{attribute_name}"
                )

        if (
                mutation_class.MUTATION_API_VERSION
                != self.API_VERSION
        ):
            raise RuntimeError(
                f"{basename} 的 MUTATION_API_VERSION "
                f"不兼容: "
                f"{mutation_class.MUTATION_API_VERSION}"
            )

        mutation_id = mutation_class.MUTATION_ID

        if not isinstance(mutation_id, str):
            raise TypeError(
                f"{basename} 的 MUTATION_ID 必须是字符串"
            )

        namespace = mutation_class.MODULE_NAMESPACE

        if (
                not isinstance(namespace, str)
                or not namespace.strip()
        ):
            raise TypeError(
                f"{basename} 的 MODULE_NAMESPACE "
                "必须是非空字符串"
            )

        for method_name in self.REQUIRED_METHODS:
            method = getattr(
                mutation_class,
                method_name,
                None,
            )

            if not callable(method):
                raise TypeError(
                    f"{basename} 缺少可调用方法: "
                    f"{method_name}"
                )

        return mutation_class, module

    def detect_for_keys(
            self,
            keys,
            source_name,
    ):
        keys = list(keys)
        matches = []

        for mutation_id, mutation_class in (
                self.mutations.items()
        ):
            try:
                score = mutation_class.detect(keys)
            except Exception as exception:
                print(
                    "⚠️ [AnimaBaker] Mutation 检测失败: "
                    f"{mutation_id}, "
                    f"source={source_name}, "
                    f"error={exception}"
                )
                continue

            if isinstance(score, bool):
                score = 100 if score else 0

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
            raise RuntimeError(
                "无法唯一判断 Mutation：\n"
                f"  来源: {source_name}\n"
                f"  候选: "
                f"{[item[1] for item in best_matches]}\n"
                f"  得分: {best_score}"
            )

        score, mutation_id, mutation_class = (
            best_matches[0]
        )

        print(
            "🧬 [AnimaBaker] 检测到 Mutation: "
            f"{mutation_id} | "
            f"source={source_name} | "
            f"score={score}"
        )

        return mutation_class


def _detect_lora_mutation_v4_only(
        mutation_registry,
        lora_keys,
        source_name,
):
    """
    LoRA 只允许最新 V4 Mutation。

    注意：
    此限制只作用于 LoRA，不作用于底模 checkpoint。
    """

    lora_keys = list(lora_keys)

    detected = mutation_registry.detect_for_keys(
        lora_keys,
        source_name,
    )

    contains_current_graft = any(
        _contains_current_graft_key(key)
        for key in lora_keys
    )

    if detected is not None:
        detected_id = str(
            detected.MUTATION_ID
        )

        if detected_id != ANIMA_GRAFTED4_MUTATION_ID:
            raise RuntimeError(
                "检测到旧版 Mutation LoRA，当前烧录器禁止 "
                "V2/V3 Mutation LoRA 混用：\n"
                f"  LoRA: {source_name}\n"
                f"  检测到: {detected_id}\n"
                f"  只允许: {ANIMA_GRAFTED4_MUTATION_ID}\n\n"
                "该限制只针对 LoRA，不针对底模。"
            )

        return detected

    if not contains_current_graft:
        return None

    mutation_class = (
        mutation_registry.mutations.get(
            ANIMA_GRAFTED4_MUTATION_ID
        )
    )

    if mutation_class is None:
        raise RuntimeError(
            "检测到 Grafted4 LoRA 参数，但缺少 Mutation 文件：\n"
            f"  Mutation/{ANIMA_GRAFTED4_MUTATION_ID}.py"
        )

    print(
        "🧬 [AnimaBaker] 根据 Grafted4 命名空间识别 V4 LoRA: "
        f"{source_name}"
    )

    return mutation_class


def _resolve_checkpoint_and_lora_mutation(
        checkpoint_mutation_class,
        lora_mutation_classes,
):
    active_lora_mutations = [
        mutation_class
        for mutation_class in lora_mutation_classes
        if mutation_class is not None
    ]

    if not active_lora_mutations:
        return checkpoint_mutation_class

    lora_ids = {
        str(mutation_class.MUTATION_ID)
        for mutation_class in active_lora_mutations
    }

    if lora_ids != {
        ANIMA_GRAFTED4_MUTATION_ID
    }:
        raise RuntimeError(
            "内部错误：LoRA Mutation 过滤后仍存在非 V4 架构："
            f"{sorted(lora_ids)}"
        )

    selected = active_lora_mutations[0]

    if checkpoint_mutation_class is None:
        return selected

    checkpoint_id = str(
        checkpoint_mutation_class.MUTATION_ID
    )

    if checkpoint_id != ANIMA_GRAFTED4_MUTATION_ID:
        raise RuntimeError(
            "当前底模 Mutation 架构与 V4 Mutation LoRA "
            "不兼容：\n"
            f"  底模 Mutation: {checkpoint_id}\n"
            f"  LoRA Mutation: {ANIMA_GRAFTED4_MUTATION_ID}\n\n"
            "旧 Mutation 底模本身仍然可以加载；"
            "但是不能直接向旧架构底模应用 V4 架构 LoRA。"
        )

    return selected


def _extract_mutation_tensors(
        state_dict,
        mutation_class,
):
    if mutation_class is None:
        return {}

    result = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        try:
            is_mutation = (
                mutation_class.is_mutation_key(key)
            )
        except Exception:
            is_mutation = False

        if is_mutation:
            result[key] = value

    return result


def _load_grafted4_sidecar_config(model_path):
    """Load optional topology metadata saved next to a Grafted4 checkpoint."""
    model_path = os.path.abspath(str(model_path))
    candidates = (
        os.path.join(
            os.path.dirname(model_path),
            "image_graft_config_v4.json",
        ),
        os.path.splitext(model_path)[0] + ".json",
    )

    for config_path in candidates:
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exception:
            raise RuntimeError(
                "无法读取 Grafted4 架构配置："
                f"{config_path}\n{exception}"
            ) from exception

        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "Grafted4 架构配置必须是 JSON object："
                f"{config_path}"
            )

        if not any(
            key in payload
            for key in (
                "mudd_config",
                "attnres_config",
                "mor_config",
                "extra_block_config",
            )
        ):
            continue

        print(
            "🧬 [AnimaBaker] 已读取 Grafted4 架构配置: "
            f"{config_path}"
        )
        return dict(payload)

    return None


def _load_mutation_base_weights(
        diffusion_model,
        mutation_class,
        source_state_dict,
        require_complete=False,
):
    if not source_state_dict:
        if require_complete:
            raise RuntimeError(
                "底模被识别为 Mutation 模型，"
                "但没有提取到 Mutation 参数。"
            )

        return 0

    target_state_dict = diffusion_model.state_dict()
    target_keys = set(target_state_dict.keys())

    target_mutation_keys = {
        key
        for key in target_keys
        if mutation_class.is_mutation_key(key)
    }

    load_state_dict = {}
    shape_mismatches = []
    unmapped_keys = []

    for source_key, source_tensor in (
            source_state_dict.items()
    ):
        try:
            if not mutation_class.is_mutation_key(
                    source_key
            ):
                continue
        except Exception:
            continue

        _validate_tensor_finite(
            source_tensor,
            source_key,
        )

        target_key = _find_target_key_by_suffix(
            source_key,
            target_keys,
        )

        if target_key is None:
            unmapped_keys.append(source_key)
            continue

        target_tensor = target_state_dict[
            target_key
        ]

        if (
                tuple(source_tensor.shape)
                != tuple(target_tensor.shape)
        ):
            shape_mismatches.append(
                (
                    source_key,
                    tuple(source_tensor.shape),
                    target_key,
                    tuple(target_tensor.shape),
                )
            )
            continue

        load_state_dict[target_key] = (
            source_tensor
        )

    if shape_mismatches:
        preview = "\n".join(
            f"  - {source_key} {source_shape} -> "
            f"{target_key} {target_shape}"
            for (
                source_key,
                source_shape,
                target_key,
                target_shape,
            ) in shape_mismatches[:30]
        )

        raise RuntimeError(
            "Mutation 参数形状不匹配：\n"
            f"{preview}"
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
            "底模 Mutation 参数存在无法映射的键：\n"
            f"{preview}\n"
            "已停止加载，避免随机初始化层影响生图。"
        )

    if require_complete and missing_target_keys:
        preview = "\n".join(
            f"  - {key}"
            for key in missing_target_keys[:30]
        )

        raise RuntimeError(
            "安装后的 Mutation 架构缺少底模权重：\n"
            f"{preview}\n"
            "已停止加载，避免随机初始化层影响生图。"
        )

    if require_complete and not load_state_dict:
        raise RuntimeError(
            "底模包含 Mutation 参数，"
            "但安装架构后没有任何参数能够映射。"
        )

    incompatible = diffusion_model.load_state_dict(
        load_state_dict,
        strict=False,
    )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Mutation 二次加载出现 unexpected keys：\n"
            + "\n".join(
                f"  - {key}"
                for key in (
                    incompatible.unexpected_keys[:30]
                )
            )
        )

    print(
        "✅ [AnimaBaker] 已加载底模 Mutation 参数: "
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

    if not isinstance(patches, Mapping):
        return 0

    count = 0

    for key in patches.keys():
        try:
            if mutation_class.is_mutation_key(
                    str(key)
            ):
                count += 1
        except Exception:
            continue

    return count


# ============================================================
# LoRA 键解析
# ============================================================

_LORA_SUFFIX_RE = re.compile(
    r"""
    ^
    (?P<base>.*?)
    \.
    (?P<adapter>
        lora_A
        |lora_B
        |lora_down
        |lora_up
        |lora_mid
        |lora_magnitude_vector
        |dora_scale

        |diff
        |diff_b
        |diff_norm
        |weight_decompose

        |hada_w1_a
        |hada_w1_b
        |hada_w2_a
        |hada_w2_b
        |hada_t1
        |hada_t2

        |lokr_w1
        |lokr_w2
        |lokr_w1_a
        |lokr_w1_b
        |lokr_w2_a
        |lokr_w2_b
        |lokr_t1
        |lokr_t2

        |ia3_lora
        |ia3_w

        |oft_blocks
        |boft_blocks
        |oft_diag
        |boft_diag

        |alpha
        |scale
    )
    (?P<tail>
        (?:\.weight)?
        (?:\.[0-9]+)?
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _split_lora_adapter_key(
        key: str,
) -> Optional[Tuple[str, str]]:
    key = str(key).replace("/", ".")

    match = _LORA_SUFFIX_RE.match(key)

    if match is None:
        return None

    return (
        match.group("base"),
        "." + match.group("adapter")
        + match.group("tail"),
    )


def _normalize_lora_suffix(
        suffix: str,
) -> str:
    lowered = suffix.lower()

    if lowered.startswith(".lora_a"):
        return (
            ".lora_down"
            + suffix[len(".lora_A"):]
        )

    if lowered.startswith(".lora_b"):
        return (
            ".lora_up"
            + suffix[len(".lora_B"):]
        )

    return suffix


def _convert_peft_lora_keys(
        lora_state_dict,
):
    result = {}
    converted_count = 0

    for raw_key, tensor in (
            lora_state_dict.items()
    ):
        key = str(raw_key)

        split = _split_lora_adapter_key(key)

        if split is None:
            output_key = key
        else:
            base, suffix = split

            output_key = (
                base
                + _normalize_lora_suffix(suffix)
            )

            if output_key != key:
                converted_count += 1

        if output_key in result:
            raise RuntimeError(
                "PEFT 键转换产生重复目标键：\n"
                f"  {output_key}"
            )

        result[output_key] = tensor

    return result, converted_count


# ============================================================
# 严格、确定性的路径映射
# ============================================================

def _model_path_aliases(
        actual_base: str,
) -> Set[str]:
    """
    只从真实模型路径生成确定性别名。

    不执行 '_' -> '.' 猜测。

    例如真实路径：
        blocks.3.mudd_graft.current_memory_proj

    生成：
        blocks.3.mudd_graft.current_memory_proj
        blocks_3_mudd_graft_current_memory_proj

    如果两个真实路径产生同一个扁平别名，后续会进行形状消歧；
    仍不能唯一确定时直接报错。
    """

    raw = str(actual_base).replace(
        "/",
        ".",
    ).strip(".")

    stripped = _strip_repeated_dotted_prefixes(
        raw
    )

    variants = {
        raw.lower(),
        stripped.lower(),
    }

    if "refiner.blocks" in stripped:
        variants.add(
            stripped.replace(
                "refiner.blocks",
                "refiner_blocks",
            ).lower()
        )

    if "refiner_blocks" in stripped:
        variants.add(
            stripped.replace(
                "refiner_blocks",
                "refiner.blocks",
            ).lower()
        )

    result = set()

    for value in variants:
        value = value.strip("._")

        if not value:
            continue

        result.add(value)
        result.add(value.replace(".", "_"))

    return result


def _source_path_aliases(
        source_base: str,
) -> Set[str]:
    """
    解析 LoRA 来源路径。

    只做：
      1. 删除明确的模型前缀；
      2. 保留来源的点/下划线结构；
      3. 点路径可以生成对应扁平形式。

    不把来源中的下划线反向猜测成点。
    """

    raw = str(source_base).replace(
        "/",
        ".",
    ).strip("._")

    stripped_lycoris, _ = (
        _strip_lycoris_model_prefix(raw)
    )

    stripped_dotted = (
        _strip_repeated_dotted_prefixes(
            stripped_lycoris
        )
    )

    variants = {
        raw.lower(),
        stripped_lycoris.lower(),
        stripped_dotted.lower(),
    }

    result = set()

    for value in variants:
        value = value.strip("._")

        if not value:
            continue

        result.add(value)

        if "." in value:
            result.add(
                value.replace(".", "_")
            )

    return result


def _build_actual_model_index(
        model_patcher,
):
    state_dict = _get_state_dict_from_owner(
        model_patcher
    )

    alias_index: Dict[str, List[str]] = {}

    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            continue

        if not torch.is_tensor(tensor):
            continue

        if not (
                key.endswith(".weight")
                or key.endswith(".bias")
        ):
            continue

        base = _remove_parameter_suffix(key)

        for alias in _model_path_aliases(base):
            values = alias_index.setdefault(
                alias,
                [],
            )

            if key not in values:
                values.append(key)

    return state_dict, alias_index


def _find_lora_tensor(
        lora_state_dict,
        source_base,
        suffixes,
):
    for suffix in suffixes:
        value = lora_state_dict.get(
            source_base + suffix
        )

        if torch.is_tensor(value):
            return value

    return None


def _lora_pair_shapes(
        lora_state_dict,
        source_base,
):
    down = _find_lora_tensor(
        lora_state_dict,
        source_base,
        (
            ".lora_down.weight",
            ".lora_A.weight",
            ".lora_down",
            ".lora_A",
        ),
    )

    up = _find_lora_tensor(
        lora_state_dict,
        source_base,
        (
            ".lora_up.weight",
            ".lora_B.weight",
            ".lora_up",
            ".lora_B",
        ),
    )

    return down, up


def _target_matches_lora_shape(
        target_tensor,
        down,
        up,
):
    if not torch.is_tensor(target_tensor):
        return False

    if down is None or up is None:
        return True

    if down.ndim < 2 or up.ndim < 2:
        return True

    if target_tensor.ndim == 2:
        return (
            down.shape[-1]
            == target_tensor.shape[1]
            and up.shape[0]
            == target_tensor.shape[0]
            and down.shape[0]
            == up.shape[1]
        )

    if target_tensor.ndim >= 3:
        return (
            down.shape[1]
            == target_tensor.shape[1]
            and up.shape[0]
            == target_tensor.shape[0]
        )

    return True


def _resolve_lora_target_key(
        source_base,
        lora_state_dict,
        model_state_dict,
        alias_index,
        lora_name,
):
    candidates = []

    for alias in _source_path_aliases(
            source_base
    ):
        candidates.extend(
            alias_index.get(alias, [])
        )

    candidates = list(
        dict.fromkeys(candidates)
    )

    if not candidates:
        return None

    down, up = _lora_pair_shapes(
        lora_state_dict,
        source_base,
    )

    shape_matches = [
        candidate
        for candidate in candidates
        if _target_matches_lora_shape(
            model_state_dict.get(candidate),
            down,
            up,
        )
    ]

    shape_matches = list(
        dict.fromkeys(shape_matches)
    )

    if len(shape_matches) == 1:
        return shape_matches[0]

    if not shape_matches:
        raise RuntimeError(
            "LoRA 路径能够找到模型候选层，但形状全部不匹配：\n"
            f"  LoRA: {lora_name}\n"
            f"  source: {source_base}\n"
            f"  candidates: {candidates[:20]}"
        )

    weight_matches = [
        candidate
        for candidate in shape_matches
        if candidate.endswith(".weight")
    ]

    if len(weight_matches) == 1:
        return weight_matches[0]

    raise RuntimeError(
        "LoRA 非规范扁平命名产生了多个可能目标层，"
        "为防止静默映射错层，已停止加载：\n"
        f"  LoRA: {lora_name}\n"
        f"  source: {source_base}\n"
        "  候选层：\n"
        + "\n".join(
            f"    - {candidate}"
            for candidate in shape_matches[:30]
        )
        + "\n\n"
        "请在训练或导出 LoRA 时保留完整模块路径，"
        "或者确保扁平命名在模型中唯一。"
    )


def _call_lora_key_builder(
        function,
        owner,
        key_map,
):
    if not callable(function):
        return key_map

    attempts = (
        (owner, key_map),
        (owner,),
    )

    for arguments in attempts:
        try:
            result = function(*arguments)
        except TypeError:
            continue
        except Exception:
            continue

        if isinstance(result, Mapping):
            key_map.update(result)
            break

    return key_map


def _validate_native_model_mapping(
        source_base,
        native_target,
        model_state_dict,
        lora_state_dict,
):
    target_key = _mapping_target_to_string(
        native_target
    )

    if target_key is None:
        return False

    target_tensor = model_state_dict.get(
        target_key
    )

    if target_tensor is None:
        return False

    down, up = _lora_pair_shapes(
        lora_state_dict,
        source_base,
    )

    return _target_matches_lora_shape(
        target_tensor,
        down,
        up,
    )


def _build_comfy_lora_key_map(
        model_patcher,
        clip_obj,
        lora_state_dict,
        lora_name,
):
    key_map = {}

    model_owner = getattr(
        model_patcher,
        "model",
        None,
    )

    key_map = _call_lora_key_builder(
        getattr(
            comfy.lora,
            "model_lora_keys_unet",
            None,
        ),
        model_owner,
        key_map,
    )

    clip_model = getattr(
        clip_obj,
        "cond_stage_model",
        None,
    )

    if clip_model is not None:
        key_map = _call_lora_key_builder(
            getattr(
                comfy.lora,
                "model_lora_keys_clip",
                None,
            ),
            clip_model,
            key_map,
        )

    model_state_dict, alias_index = (
        _build_actual_model_index(
            model_patcher
        )
    )

    adapter_bases = []

    for key in lora_state_dict.keys():
        split = _split_lora_adapter_key(
            str(key)
        )

        if split is None:
            continue

        base = split[0]

        if base not in adapter_bases:
            adapter_bases.append(base)

    mapped_model_bases = {}
    unresolved_model_bases = []

    for source_base in adapter_bases:
        resolved_target = (
            _resolve_lora_target_key(
                source_base=source_base,
                lora_state_dict=lora_state_dict,
                model_state_dict=model_state_dict,
                alias_index=alias_index,
                lora_name=lora_name,
            )
        )

        if resolved_target is not None:
            key_map[source_base] = (
                resolved_target
            )

            mapped_model_bases[source_base] = (
                resolved_target
            )
            continue

        native_target = key_map.get(
            source_base
        )

        if native_target is not None:
            if _validate_native_model_mapping(
                    source_base,
                    native_target,
                    model_state_dict,
                    lora_state_dict,
            ):
                mapped_model_bases[
                    source_base
                ] = _mapping_target_to_string(
                    native_target
                )
                continue

        if _is_explicit_model_adapter_base(
                source_base
        ):
            unresolved_model_bases.append(
                source_base
            )

    if unresolved_model_bases:
        raise RuntimeError(
            "LoRA 中存在无法严格映射到当前扩散模型的路径：\n"
            f"  LoRA: {lora_name}\n"
            f"  数量: {len(unresolved_model_bases)}\n"
            + "\n".join(
                f"    - {key}"
                for key in (
                    unresolved_model_bases[:30]
                )
            )
            + "\n\n"
            "本版本不会使用全局下划线转点号的模糊匹配，"
            "因为这可能把 LoRA 静默应用到错误层。"
        )

    print(
        "🔗 [AnimaBaker] LoRA 严格映射完成: "
        f"{lora_name} | "
        f"MODEL={len(mapped_model_bases)}, "
        f"key_map={len(key_map)}, "
        f"adapter_base={len(adapter_bases)}"
    )

    for source_base, target_key in (
            list(mapped_model_bases.items())[:10]
    ):
        print(
            f"    {source_base} -> {target_key}"
        )

    return key_map, mapped_model_bases


def _load_lora_with_explicit_key_map(
        model_patcher,
        clip_obj,
        lora_state_dict,
        model_strength,
        clip_strength,
        lora_name,
):
    converted_sd, converted_count = (
        _convert_peft_lora_keys(
            lora_state_dict
        )
    )

    key_map, mapped_model_bases = (
        _build_comfy_lora_key_map(
            model_patcher=model_patcher,
            clip_obj=clip_obj,
            lora_state_dict=converted_sd,
            lora_name=lora_name,
        )
    )

    try:
        loaded = comfy.lora.load_lora(
            converted_sd,
            key_map,
        )
    except Exception as exception:
        raise RuntimeError(
            f"解析 LoRA 失败：{lora_name}\n"
            f"原因: {exception}"
        ) from exception

    if not isinstance(loaded, Mapping):
        raise RuntimeError(
            "comfy.lora.load_lora() "
            "没有返回有效 patch 字典：\n"
            f"  {lora_name}"
        )

    model_result = model_patcher.clone()

    loaded_model_keys = model_result.add_patches(
        loaded,
        float(model_strength),
    )

    if loaded_model_keys is None:
        loaded_model_keys = []

    clip_result = clip_obj
    loaded_clip_keys = []

    clip_patcher = _get_clip_patcher(
        clip_obj
    )

    if clip_patcher is not None:
        if hasattr(clip_obj, "clone"):
            clip_result = clip_obj.clone()
        else:
            clip_result = copy.copy(clip_obj)
            clip_result.patcher = (
                clip_patcher.clone()
            )

        result_clip_patcher = (
            _get_clip_patcher(clip_result)
        )

        if result_clip_patcher is not None:
            loaded_clip_keys = (
                result_clip_patcher.add_patches(
                    loaded,
                    float(clip_strength),
                )
            )

            if loaded_clip_keys is None:
                loaded_clip_keys = []

    loaded_model_set = {
        str(key)
        for key in loaded_model_keys
    }

    loaded_clip_set = {
        str(key)
        for key in loaded_clip_keys
    }

    loaded_patch_keys = {
        str(key)
        for key in loaded.keys()
    }

    actually_loaded = (
        loaded_model_set
        | loaded_clip_set
    )

    not_applied = sorted(
        loaded_patch_keys
        - actually_loaded
    )

    expected_model_targets = {
        str(target)
        for target in mapped_model_bases.values()
        if target is not None
    }

    unapplied_model_targets = sorted(
        expected_model_targets
        & set(not_applied)
    )

    if unapplied_model_targets:
        raise RuntimeError(
            "LoRA 已映射为扩散模型 patch，"
            "但 ModelPatcher 没有接收：\n"
            f"  LoRA: {lora_name}\n"
            + "\n".join(
                f"    - {key}"
                for key in (
                    unapplied_model_targets[:30]
                )
            )
        )

    if not_applied:
        print(
            "⚠️ [AnimaBaker] 有 "
            f"{len(not_applied)} 个辅助 patch 未被接收。"
        )

    if (
            mapped_model_bases
            and abs(float(model_strength)) > 0.0001
            and not loaded_model_set
    ):
        raise RuntimeError(
            f"LoRA {lora_name} 包含扩散模型权重，"
            "但没有任何 MODEL patch 被加载。"
        )

    print(
        "✅ [AnimaBaker] LoRA patch 加载完成: "
        f"{lora_name} | "
        f"MODEL={len(loaded_model_set)}, "
        f"CLIP={len(loaded_clip_set)}, "
        f"PEFT转换={converted_count}"
    )

    return (
        model_result,
        clip_result,
        converted_sd,
    )


# ============================================================
# FP32 高精度烧录
# ============================================================

def _is_oom_exception(
        exception,
):
    if isinstance(
            exception,
            torch.OutOfMemoryError,
    ):
        return True

    message = str(exception).lower()

    return (
        "out of memory" in message
        or "allocation on device" in message
        or "not enough memory" in message
    )


def _calculate_patch_weight_fp32(
        patch_list,
        source_tensor,
        key,
        calc_device,
):
    """
    基础权重和 LoRA/LyCORIS 内部张量均以 FP32 参与计算。

    comfy.lora.calculate_weight() 会按照传入 weight 的 dtype/device
    转换 patch 张量，因此传入 FP32 weight 可以避免 BF16 LoRA
    直接进行低精度矩阵计算。
    """

    preferred_device = torch.device(
        calc_device
    )

    def calculate_on(device):
        weight_fp32 = source_tensor.detach().to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )

        result = comfy.lora.calculate_weight(
            patch_list,
            weight_fp32,
            key,
        )

        if not torch.is_tensor(result):
            raise RuntimeError(
                f"calculate_weight({key}) "
                "没有返回 Tensor"
            )

        if result.dtype != torch.float32:
            result = result.to(
                dtype=torch.float32
            )

        return result

    try:
        return calculate_on(
            preferred_device
        )
    except Exception as exception:
        if (
                preferred_device.type == "cpu"
                or not _is_oom_exception(exception)
        ):
            raise

        print(
            "⚠️ [AnimaBaker] FP32 单层计算显存不足，"
            f"回退 CPU：{key}"
        )

        _safe_gc()

        return calculate_on(
            torch.device("cpu")
        )


@torch.inference_mode()
def _bake_patcher_weights(
        patcher,
        bake_dtype,
        calc_device,
        name,
):
    """
    高精度烧录流程：

    1. 模型移动到 CPU；
    2. 将浮点 Parameter/Buffer 升到 FP32 工作精度；
    3. 所有 LoRA/LyCORIS patch 在 FP32 中合并；
    4. 合并完成后统一转换一次到最终 dtype；
    5. 清除 patches/backup。

    这里故意不在烧录前转换为最终 BF16/FP16。
    """

    if patcher is None:
        return patcher

    if bake_dtype not in SUPPORTED_BAKE_DTYPES:
        raise ValueError(
            f"{name} 不支持的 bake_dtype: "
            f"{bake_dtype}"
        )

    model_inner = getattr(
        patcher,
        "model",
        None,
    )

    if model_inner is None:
        raise RuntimeError(
            f"{name} patcher 中不存在 model"
        )

    patches = getattr(
        patcher,
        "patches",
        {},
    )

    if patches is None:
        patches = {}

    if not isinstance(patches, Mapping):
        raise RuntimeError(
            f"{name} patcher.patches 不是 Mapping"
        )

    print(
        "🔥 [AnimaBaker] 开始高精度烧录 "
        f"{name} | calc={calc_device} | "
        f"workspace=float32 | "
        f"output={_dtype_name(bake_dtype)}"
    )

    model_inner.to(
        device=torch.device("cpu")
    )

    _safe_gc()

    before_counts = (
        _module_floating_dtype_counts(
            model_inner
        )
    )

    print(
        f"ℹ️ [AnimaBaker] {name} 原始 dtype: "
        f"{_format_dtype_counts(before_counts)}"
    )

    # 关键修复：
    # 先升到 FP32 工作模型，不提前降低到底层保存 dtype。
    model_inner.to(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _safe_gc()

    workspace_counts = (
        _module_floating_dtype_counts(
            model_inner
        )
    )

    if _module_needs_dtype_normalization(
            model_inner,
            torch.float32,
    ):
        raise RuntimeError(
            f"{name} 无法建立纯 FP32 工作模型："
            f"{_format_dtype_counts(workspace_counts)}"
        )

    state_dict = model_inner.state_dict()
    state_keys = set(state_dict.keys())

    unused_patch_keys = sorted(
        set(patches.keys())
        - state_keys
    )

    if unused_patch_keys:
        raise RuntimeError(
            f"{name} 中存在无法对应模型参数的 patch：\n"
            + "\n".join(
                f"  - {key}"
                for key in unused_patch_keys[:30]
            )
        )

    patch_items = list(
        patches.items()
    )

    total = len(patch_items)

    for index, (
            key,
            patch_list,
    ) in enumerate(patch_items):
        if index % 50 == 0:
            print(
                f"  - {name}: "
                f"{index + 1}/{total}"
            )

        source_tensor = state_dict.get(key)

        if source_tensor is None:
            raise RuntimeError(
                f"{name} 找不到 patch 目标：{key}"
            )

        if not torch.is_tensor(source_tensor):
            raise RuntimeError(
                f"{name} patch 目标不是 Tensor：{key}"
            )

        if not source_tensor.is_floating_point():
            raise RuntimeError(
                "LoRA patch 指向非浮点参数：\n"
                f"  {key}"
            )

        if source_tensor.dtype != torch.float32:
            raise RuntimeError(
                "FP32 工作模型中出现非 FP32 patch 目标：\n"
                f"  key={key}\n"
                f"  dtype={source_tensor.dtype}"
            )

        weight = _calculate_patch_weight_fp32(
            patch_list=patch_list,
            source_tensor=source_tensor,
            key=key,
            calc_device=calc_device,
        )

        if (
                tuple(weight.shape)
                != tuple(source_tensor.shape)
        ):
            raise RuntimeError(
                "LoRA/LyCORIS 烧录结果形状不一致：\n"
                f"  key={key}\n"
                f"  result={tuple(weight.shape)}\n"
                f"  target={tuple(source_tensor.shape)}\n\n"
                "可能原因：\n"
                "1. LoRA 映射到了错误层；\n"
                "2. grouped/depthwise Conv LoCon 不兼容；\n"
                "3. 当前 ComfyUI 不支持该 LoKr/LoHa 格式。"
            )

        if not torch.isfinite(weight).all():
            raise RuntimeError(
                f"烧录 {name} 时出现 NaN/Inf：\n"
                f"  {key}"
            )

        source_tensor.copy_(
            weight.to(
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )

        del weight

        if (
                index > 0
                and index % 100 == 0
        ):
            _safe_gc()

    _validate_module_parameters_finite(
        model_inner,
        f"{name} FP32 工作模型",
    )

    # 所有合并完成后只执行一次最终量化。
    if bake_dtype != torch.float32:
        if bake_dtype == torch.float16:
            fp16_limit = torch.finfo(
                torch.float16
            ).max

            for parameter_name, parameter in (
                    model_inner.named_parameters(
                        recurse=True
                    )
            ):
                if not parameter.is_floating_point():
                    continue

                if parameter.numel() == 0:
                    continue

                max_value = (
                    parameter.detach()
                    .abs()
                    .max()
                    .item()
                )

                if max_value > fp16_limit:
                    raise RuntimeError(
                        "参数无法安全转换为 float16：\n"
                        f"  key={parameter_name}\n"
                        f"  abs_max={max_value}\n"
                        f"  float16_max={fp16_limit}\n"
                        "请使用 bfloat16 或 float32。"
                    )

        model_inner.to(
            device=torch.device("cpu"),
            dtype=bake_dtype,
        )

        _safe_gc()

    _validate_module_parameters_finite(
        model_inner,
        name,
    )

    final_counts = (
        _module_floating_dtype_counts(
            model_inner
        )
    )

    if _module_needs_dtype_normalization(
            model_inner,
            bake_dtype,
    ):
        raise RuntimeError(
            f"{name} 最终仍存在混合 dtype："
            f"{_format_dtype_counts(final_counts)}\n"
            f"目标 dtype: {_dtype_name(bake_dtype)}"
        )

    _set_runtime_dtype_metadata(
        patcher,
        model_inner,
        bake_dtype,
    )

    patcher.patches = {}
    patcher.backup = {}

    del state_dict

    _safe_gc()

    print(
        f"✅ [AnimaBaker] {name} 烧录完成 | "
        f"最终 dtype: "
        f"{_format_dtype_counts(final_counts)}"
    )

    return patcher


# ============================================================
# 烧录核心
# ============================================================

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
                    abs(float(lora[1])) > 0.0001
                    or abs(float(lora[2])) > 0.0001
                )
            )
        ]

        print(
            "🧹 [AnimaBaker] 清理残留模型和显存..."
        )

        comfy.model_management.unload_all_models()
        _safe_gc()

        target_device = _device_from_string(
            device
        )

        print(
            "🔧 [AnimaBaker] LoRA FP32 计算设备: "
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
                f"找不到 VAE 文件: {vae}"
            )

        clip_path, _ = _find_model_path(
            clip,
            ["clip"],
        )

        if clip_path is None:
            raise RuntimeError(
                f"找不到 CLIP 文件: {clip}"
            )

        mutation_registry = None
        checkpoint_mutation_class = None
        selected_mutation_class = None
        checkpoint_scan_sd = None
        mutation_base_tensors = {}

        lora_mutation_map = {}
        lora_key_cache = {}

        # ----------------------------------------------------
        # Mutation 检测
        # ----------------------------------------------------

        if enable_mutation:
            mutation_directory = os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
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
                    "🔍 [AnimaBaker] 扫描底模 Mutation..."
                )

                checkpoint_scan_sd = (
                    comfy.utils.load_torch_file(
                        model_path,
                        safe_load=True,
                    )
                )

                if not isinstance(
                        checkpoint_scan_sd,
                        Mapping,
                ):
                    raise RuntimeError(
                        "底模文件没有返回有效 state_dict"
                    )

                # 底模允许注册表中的旧 Mutation。
                checkpoint_mutation_class = (
                    mutation_registry.detect_for_keys(
                        checkpoint_scan_sd.keys(),
                        source_name=(
                            f"checkpoint:{model}"
                        ),
                    )
                )

                lora_mutation_classes = []

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
                            f"找不到 LoRA: {lora_name}"
                        )

                    print(
                        "🔍 [AnimaBaker] 扫描 LoRA Mutation: "
                        f"{lora_name}"
                    )

                    lora_scan_sd = (
                        comfy.utils.load_torch_file(
                            lora_path,
                            safe_load=True,
                        )
                    )

                    if not isinstance(
                            lora_scan_sd,
                            Mapping,
                    ):
                        raise RuntimeError(
                            "LoRA 文件没有返回有效 state_dict："
                            f"{lora_name}"
                        )

                    lora_keys = list(
                        lora_scan_sd.keys()
                    )

                    lora_key_cache[
                        lora_name
                    ] = lora_keys

                    # LoRA 只允许最新 V4。
                    lora_mutation_class = (
                        _detect_lora_mutation_v4_only(
                            mutation_registry,
                            lora_keys,
                            source_name=(
                                f"lora:{lora_name}"
                            ),
                        )
                    )

                    lora_mutation_map[
                        lora_name
                    ] = lora_mutation_class

                    lora_mutation_classes.append(
                        lora_mutation_class
                    )

                    del lora_scan_sd
                    _safe_gc()

                selected_mutation_class = (
                    _resolve_checkpoint_and_lora_mutation(
                        checkpoint_mutation_class,
                        lora_mutation_classes,
                    )
                )

                if checkpoint_mutation_class is not None:
                    mutation_base_tensors = (
                        _extract_mutation_tensors(
                            checkpoint_scan_sd,
                            checkpoint_mutation_class,
                        )
                    )

                del checkpoint_scan_sd
                checkpoint_scan_sd = None
                _safe_gc()

                if selected_mutation_class is None:
                    print(
                        "ℹ️ [AnimaBaker] 未检测到 Mutation"
                    )
                else:
                    print(
                        "🧬 [AnimaBaker] 最终 Mutation: "
                        f"{selected_mutation_class.MUTATION_ID}"
                    )
            else:
                print(
                    "ℹ️ [AnimaBaker] Mutation 目录中没有可用架构"
                )

        # ----------------------------------------------------
        # 加载底模
        # ----------------------------------------------------

        print(
            "📥 [AnimaBaker] 加载 Anima 底模: "
            f"{model_path}"
        )

        ckpt_out = (
            comfy.sd.load_checkpoint_guess_config(
                model_path,
                output_vae=False,
                output_clip=False,
            )
        )

        model_obj = ckpt_out[0]

        # ----------------------------------------------------
        # 加载 VAE
        # ----------------------------------------------------

        print(
            "📥 [AnimaBaker] 加载 VAE: "
            f"{vae_path}"
        )

        vae_sd = comfy.utils.load_torch_file(
            vae_path
        )

        vae_obj = comfy.sd.VAE(
            sd=vae_sd
        )

        del vae_sd
        _safe_gc()

        # ----------------------------------------------------
        # 加载 CLIP
        # ----------------------------------------------------

        print(
            "📥 [AnimaBaker] 加载 Qwen3 CLIP: "
            f"{clip_path}"
        )

        if (
                AnimaTEModel is None
                or AnimaTokenizer is None
        ):
            raise RuntimeError(
                "当前 ComfyUI 中找不到 "
                "AnimaTEModel / AnimaTokenizer"
            )

        clip_sd = comfy.utils.load_torch_file(
            clip_path,
            safe_load=True,
        )

        clip_target = (
            comfy.supported_models_base.ClipTarget(
                tokenizer=AnimaTokenizer,
                clip=AnimaTEModel,
            )
        )

        clip_obj = CLIP(
            clip_target,
            embedding_directory=None,
        )

        clip_obj.load_sd(clip_sd)

        del clip_sd
        _safe_gc()

        model_clone = model_obj.clone()

        if hasattr(clip_obj, "clone"):
            clip_clone = clip_obj.clone()
        else:
            clip_clone = copy.copy(clip_obj)

        diffusion_model = (
            _find_diffusion_model_from_patcher(
                model_clone
            )
        )

        if diffusion_model is None:
            raise RuntimeError(
                "无法从 ModelPatcher 中找到 diffusion_model"
            )

        model_parameter_dtype = _module_dtype(
            diffusion_model,
            torch.bfloat16,
        )

        model_base_dtype = (
            _resolve_model_runtime_dtype(
                model_clone,
                diffusion_model,
                model_parameter_dtype,
            )
        )

        model_target_dtype = _dtype_from_string(
            save_dtype,
            model_base_dtype,
        )

        if model_target_dtype is None:
            model_target_dtype = (
                model_base_dtype
            )

        clip_patcher = _get_clip_patcher(
            clip_clone
        )

        clip_model_inner = (
            clip_patcher.model
            if clip_patcher is not None
            else None
        )

        clip_parameter_dtype = _module_dtype(
            clip_model_inner,
            torch.bfloat16,
        )

        clip_base_dtype = (
            _resolve_model_runtime_dtype(
                clip_patcher,
                clip_model_inner,
                clip_parameter_dtype,
            )
        )

        clip_target_dtype = _dtype_from_string(
            save_dtype,
            clip_base_dtype,
        )

        if clip_target_dtype is None:
            clip_target_dtype = (
                clip_base_dtype
            )

        print(
            "🔧 [AnimaBaker] 主模型 dtype: "
            f"{_dtype_name(model_base_dtype)} -> "
            f"{_dtype_name(model_target_dtype)}"
        )

        print(
            "🔧 [AnimaBaker] CLIP dtype: "
            f"{_dtype_name(clip_base_dtype)} -> "
            f"{_dtype_name(clip_target_dtype)}"
        )

        if (
                selected_mutation_class is not None
                and model_target_dtype
                == torch.float16
        ):
            print(
                "⚠️ [AnimaBaker] Mutation 模型使用 FP16，"
                "推荐改为 BF16，以降低门控/归一化溢出风险。"
            )

        # ----------------------------------------------------
        # 安装 Mutation
        # ----------------------------------------------------

        if selected_mutation_class is not None:
            print(
                "🧬 [AnimaBaker] 安装 Mutation: "
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
                    source_keys.extend(lora_keys)

            install_kwargs = {
                "source_keys": source_keys,
                "source_state_dict": mutation_base_tensors,
            }
            if (
                str(selected_mutation_class.MUTATION_ID)
                == ANIMA_GRAFTED4_MUTATION_ID
                and checkpoint_mutation_class is not None
                and str(checkpoint_mutation_class.MUTATION_ID)
                == ANIMA_GRAFTED4_MUTATION_ID
            ):
                install_kwargs["runtime_config"] = (
                    _load_grafted4_sidecar_config(model_path)
                )

            selected_mutation_class.install(
                diffusion_model,
                **install_kwargs,
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
                    "Mutation install() 没有设置正确标记：\n"
                    f"  期望: "
                    f"{selected_mutation_class.MUTATION_ID}\n"
                    f"  实际: {installed_id}"
                )

            # 只有底模本身是该 Mutation 时才要求完整加载。
            require_complete = (
                checkpoint_mutation_class
                is not None
            )

            if checkpoint_mutation_class is not None:
                checkpoint_id = str(
                    checkpoint_mutation_class
                    .MUTATION_ID
                )

                selected_id = str(
                    selected_mutation_class
                    .MUTATION_ID
                )

                if checkpoint_id != selected_id:
                    raise RuntimeError(
                        "底模 Mutation 与最终安装架构不一致：\n"
                        f"  checkpoint={checkpoint_id}\n"
                        f"  selected={selected_id}"
                    )

            _load_mutation_base_weights(
                diffusion_model,
                selected_mutation_class,
                mutation_base_tensors,
                require_complete=require_complete,
            )

            _validate_module_parameters_finite(
                diffusion_model,
                "安装 Mutation 后的扩散模型",
            )

        del mutation_base_tensors
        _safe_gc()

        # ----------------------------------------------------
        # 应用 LoRA
        # ----------------------------------------------------

        print(
            "🛠️ [AnimaBaker] LoRA 数量: "
            f"{len(active_loras)}"
        )

        for (
                lora_name,
                model_strength,
                clip_strength,
        ) in active_loras:
            lora_path, _ = _find_model_path(
                lora_name,
                ["loras"],
            )

            if lora_path is None:
                raise RuntimeError(
                    f"找不到 LoRA: {lora_name}"
                )

            print(
                "  - 应用 LoRA: "
                f"{lora_name} | "
                f"model={model_strength}, "
                f"clip={clip_strength}"
            )

            raw_lora_sd = (
                comfy.utils.load_torch_file(
                    lora_path,
                    safe_load=True,
                )
            )

            if not isinstance(
                    raw_lora_sd,
                    Mapping,
            ):
                raise RuntimeError(
                    "LoRA 文件没有返回有效 state_dict："
                    f"{lora_path}"
                )

            mutation_patch_count_before = 0

            if selected_mutation_class is not None:
                mutation_patch_count_before = (
                    _count_mutation_patches(
                        model_clone,
                        selected_mutation_class,
                    )
                )

            (
                model_clone,
                clip_clone,
                converted_lora_sd,
            ) = _load_lora_with_explicit_key_map(
                model_patcher=model_clone,
                clip_obj=clip_clone,
                lora_state_dict=raw_lora_sd,
                model_strength=model_strength,
                clip_strength=clip_strength,
                lora_name=lora_name,
            )

            lora_mutation_class = (
                lora_mutation_map.get(lora_name)
            )

            if (
                    selected_mutation_class is not None
                    and lora_mutation_class
                    is not None
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
                    mutation_keys = []

                    for key in (
                            converted_lora_sd.keys()
                    ):
                        try:
                            if (
                                selected_mutation_class
                                .is_mutation_key(key)
                            ):
                                mutation_keys.append(key)
                        except Exception:
                            continue

                    raise RuntimeError(
                        f"LoRA {lora_name} 被识别为 "
                        f"{selected_mutation_class.MUTATION_ID}，"
                        "但没有新增 Mutation patch。\n"
                        + "\n".join(
                            f"  - {key}"
                            for key in (
                                mutation_keys[:20]
                            )
                        )
                    )

                print(
                    "    ✅ Mutation patch 增加: "
                    f"{added_count}"
                )

            del raw_lora_sd
            del converted_lora_sd
            _safe_gc()

        # ----------------------------------------------------
        # 高精度烧录
        # ----------------------------------------------------

        clip_patcher = _get_clip_patcher(
            clip_clone
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

        if model_needs_bake:
            model_clone = _bake_patcher_weights(
                model_clone,
                bake_dtype=model_target_dtype,
                calc_device=target_device,
                name="UNET/Transformer",
            )
        else:
            print(
                "ℹ️ [AnimaBaker] 主模型无 patch 且 dtype 不变，"
                "跳过重写。"
            )

        if clip_needs_bake:
            clip_patcher = _get_clip_patcher(
                clip_clone
            )

            if clip_patcher is None:
                raise RuntimeError(
                    "CLIP 需要烧录，但不存在 patcher"
                )

            baked_clip_patcher = (
                _bake_patcher_weights(
                    clip_patcher,
                    bake_dtype=clip_target_dtype,
                    calc_device=target_device,
                    name="CLIP",
                )
            )

            clip_clone.patcher = (
                baked_clip_patcher
            )
        else:
            print(
                "ℹ️ [AnimaBaker] CLIP 无 patch 且 dtype 不变，"
                "保留原始参数。"
            )

        # ----------------------------------------------------
        # 最终检查和运行时修复
        # ----------------------------------------------------

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
                "主模型烧录后仍存在混合 dtype："
                f"{_format_dtype_counts(final_model_dtype_counts)}\n"
                "目标 dtype："
                f"{_dtype_name(model_target_dtype)}"
            )

        _validate_module_parameters_finite(
            model_clone.model,
            "最终主模型",
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
            diffusion_dtype = _module_dtype(
                diffusion_model,
                model_target_dtype,
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
                "x_embedder 实际 dtype 与目标不一致：\n"
                f"  x_embedder="
                f"{_dtype_name(diffusion_dtype)}\n"
                f"  target="
                f"{_dtype_name(model_target_dtype)}"
            )

        print(
            "✅ [AnimaBaker] 最终主模型 dtype: "
            f"{_dtype_name(diffusion_dtype)}"
        )

        print(
            "✅ [AnimaBaker] Parameter dtype 分布: "
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

        _safe_gc()

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


# ============================================================
# 原版 Anima 节点
# ============================================================

class SeparateModelMixerDictFuser(
    _AnimaBakerCore
):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list(
                        "checkpoints"
                    ),
                ),
                "clip": (
                    folder_paths.get_filename_list(
                        "clip"
                    ),
                ),
                "vae": (
                    folder_paths.get_filename_list(
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


# ============================================================
# Mutation Anima 节点
# ============================================================

class MutationAnimaModelBaker(
    _AnimaBakerCore
):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list(
                        "checkpoints"
                    ),
                ),
                "clip": (
                    folder_paths.get_filename_list(
                        "clip"
                    ),
                ),
                "vae": (
                    folder_paths.get_filename_list(
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
