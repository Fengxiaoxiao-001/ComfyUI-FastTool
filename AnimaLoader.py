# coding=utf-8
import os
import gc
import math
import copy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import comfy.model_management
import comfy.lora

from comfy.sd import CLIP

try:
    from comfy.text_encoders.anima import AnimaTEModel, AnimaTokenizer
except Exception:
    AnimaTEModel = None
    AnimaTokenizer = None

ANIMA_ADAPTER_FOLDER_NAME = "anima_adapters"
ANIMA_ADAPTER_DIR = os.path.join(folder_paths.models_dir, ANIMA_ADAPTER_FOLDER_NAME)
os.makedirs(ANIMA_ADAPTER_DIR, exist_ok=True)

try:
    supported_ext = folder_paths.supported_pt_extensions
except Exception:
    supported_ext = {".pt", ".pth", ".ckpt", ".safetensors"}

if ANIMA_ADAPTER_FOLDER_NAME not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[ANIMA_ADAPTER_FOLDER_NAME] = (
        [ANIMA_ADAPTER_DIR],
        supported_ext,
    )

DTYPE_CHOICES = ["auto", "from_file", "float16", "bfloat16", "float32"]


def _adapter_list():
    return ["None"] + folder_paths.get_filename_list(ANIMA_ADAPTER_FOLDER_NAME)


def _dtype_from_string(dtype, fallback=None):
    dtype_map = {
        "auto": fallback,
        "from_file": fallback,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype, fallback)


def _dtype_name(dtype):
    if dtype is None:
        return "None"
    return str(dtype).replace("torch.", "")


def _device_from_string(device):
    if device == "auto":
        return comfy.model_management.get_torch_device()

    if device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ [AnimaAdapter] CUDA 不可用，回退到 CPU")
            return torch.device("cpu")
        return comfy.model_management.get_torch_device()

    if device == "npu":
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return torch.device("npu")
        except Exception:
            pass
        print("⚠️ [AnimaAdapter] NPU 不可用，回退到自动设备")
        return comfy.model_management.get_torch_device()

    return torch.device(device)


def _module_first_param(module):
    try:
        return next(module.parameters())
    except StopIteration:
        return None


def _module_dtype(module, fallback=torch.bfloat16):
    p = _module_first_param(module)
    return p.dtype if p is not None else fallback


def _module_device(module, fallback=None):
    p = _module_first_param(module)
    if p is not None:
        return p.device
    if fallback is not None:
        return fallback
    return comfy.model_management.get_torch_device()


def _floating_dtype_from_state_dict(sd, fallback=None):
    for v in sd.values():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            return v.dtype
    return fallback


def _find_diffusion_model_from_patcher(model_patcher):
    candidates = []

    if hasattr(model_patcher, "model"):
        candidates.append(model_patcher.model)

        if hasattr(model_patcher.model, "diffusion_model"):
            candidates.append(model_patcher.model.diffusion_model)

        if hasattr(model_patcher.model, "model"):
            candidates.append(model_patcher.model.model)

            if hasattr(model_patcher.model.model, "diffusion_model"):
                candidates.append(model_patcher.model.model.diffusion_model)

    if hasattr(model_patcher, "diffusion_model"):
        candidates.append(model_patcher.diffusion_model)

    candidates.append(model_patcher)

    for obj in candidates:
        if obj is not None and hasattr(obj, "blocks"):
            return obj

    for obj in candidates:
        if obj is not None and hasattr(obj, "preprocess_text_embeds"):
            return obj

    raise RuntimeError("找不到 Anima diffusion_model，无法挂载 Adapter 插件")


def _force_anima_llm_adapter_device(diffusion_model, device=None, dtype=None):
    if diffusion_model is None or not hasattr(diffusion_model, "llm_adapter"):
        return

    llm_adapter = diffusion_model.llm_adapter

    if device is None:
        device = comfy.model_management.get_torch_device()

    if dtype is None:
        dtype = _module_dtype(diffusion_model, torch.bfloat16)

    try:
        llm_adapter.to(device=device, dtype=dtype)
    except TypeError:
        llm_adapter.to(device=device)

    try:
        if hasattr(llm_adapter, "embed"):
            llm_adapter.embed.to(device=device)
    except Exception:
        pass


def _patch_anima_preprocess_text_embeds(diffusion_model):
    if diffusion_model is None:
        return

    if not hasattr(diffusion_model, "preprocess_text_embeds"):
        return

    if getattr(diffusion_model, "_anima_device_preprocess_patched", False):
        return

    original_preprocess = diffusion_model.preprocess_text_embeds

    def patched_preprocess(self, text_embeds, text_ids, *args, **kwargs):
        target_device = (
            text_embeds.device
            if torch.is_tensor(text_embeds)
            else comfy.model_management.get_torch_device()
        )
        target_dtype = (
            text_embeds.dtype
            if torch.is_tensor(text_embeds)
            else _module_dtype(self, torch.bfloat16)
        )

        if hasattr(self, "llm_adapter"):
            try:
                self.llm_adapter.to(device=target_device, dtype=target_dtype)
            except TypeError:
                self.llm_adapter.to(device=target_device)

            try:
                if hasattr(self.llm_adapter, "embed"):
                    self.llm_adapter.embed.to(device=target_device)
            except Exception:
                pass

        if torch.is_tensor(text_ids):
            text_ids = text_ids.to(device=target_device)

        if "t5xxl_weights" in kwargs and torch.is_tensor(kwargs["t5xxl_weights"]):
            kwargs["t5xxl_weights"] = kwargs["t5xxl_weights"].to(
                device=target_device,
                dtype=target_dtype,
            )

        return original_preprocess(text_embeds, text_ids, *args, **kwargs)

    diffusion_model.preprocess_text_embeds = types.MethodType(
        patched_preprocess,
        diffusion_model,
    )
    diffusion_model._anima_device_preprocess_patched = True
    print("✅ [AnimaAdapter] 已修复 Anima preprocess_text_embeds 设备同步")


def _match_last_dim(x, target_dim):
    if x is None or not torch.is_tensor(x):
        return x

    if x.shape[-1] == target_dim:
        return x

    cur = x.shape[-1]
    if cur > target_dim:
        return x[..., :target_dim]

    pad = torch.zeros(
        *x.shape[:-1],
        target_dim - cur,
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=-1)


def _extract_clip_vision_tensor(clip_vision_output):
    if clip_vision_output is None:
        return None

    if torch.is_tensor(clip_vision_output):
        return clip_vision_output

    if isinstance(clip_vision_output, dict):
        for key in [
            "image_embeds",
            "pooled_output",
            "last_hidden_state",
            "penultimate_hidden_states",
            "cond",
        ]:
            v = clip_vision_output.get(key, None)
            if torch.is_tensor(v):
                return v

    for key in [
        "image_embeds",
        "pooled_output",
        "last_hidden_state",
        "penultimate_hidden_states",
        "cond",
    ]:
        if hasattr(clip_vision_output, key):
            v = getattr(clip_vision_output, key)
            if torch.is_tensor(v):
                return v

    return None


class ZeroLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class RuntimeSemanticScaleAdapter(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(text_dim))
        self.shift = nn.Parameter(torch.zeros(text_dim))

    def forward(self, x):
        return x * self.scale + self.shift


class RuntimeContrastModAdapter(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(text_dim))
        self.beta = nn.Parameter(torch.zeros(text_dim))

    def forward(self, x):
        return x * self.gamma + self.beta


class RuntimeColorTuneAdapter(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(text_dim))

    def forward(self, x):
        return x + self.shift


class RuntimeModTextAdapter(nn.Module):
    def __init__(self, text_dim, dit_dim):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dit_dim * 2),
            nn.SiLU(),
            ZeroLinear(dit_dim * 2),
        )

    def forward(self, text_embeds):
        global_text = text_embeds.mean(dim=1)
        gamma, beta = self.text_proj(global_text).chunk(2, dim=-1)
        return gamma, beta


class RuntimeLocalConvAdapter(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pwconv = ZeroLinear(dim)

    def forward(self, x, h=None, w=None):
        init_ndim = x.ndim

        if init_ndim == 3:
            b, n, c = x.shape

            if h is None or w is None:
                side = int(math.sqrt(n))
                if side * side != n:
                    return x
                h = side
                w = side

            hw = h * w
            if hw <= 0 or n % hw != 0:
                return x

            t = n // hw
            x = x.reshape(b, t, h, w, c)

        if x.ndim != 5:
            return x

        b, t, hh, ww, c = x.shape
        y = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, hh, ww)
        y = self.dwconv(y)
        y = y.reshape(b, t, c, hh, ww).permute(0, 1, 3, 4, 2)

        out = x + self.pwconv(y)

        if init_ndim == 3:
            out = out.reshape(b, t * h * w, c)

        return out


class RuntimeEdgeDetailConv(nn.Module):
    def __init__(self, dim):
        super().__init__()

        kernel = torch.tensor(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.register_buffer("laplacian_kernel", kernel)
        self.out = ZeroLinear(dim)

    def forward(self, x, h=None, w=None):
        init_ndim = x.ndim

        if init_ndim == 3:
            b, n, c = x.shape

            if h is None or w is None:
                side = int(math.sqrt(n))
                if side * side != n:
                    return x
                h = side
                w = side

            hw = h * w
            if hw <= 0 or n % hw != 0:
                return x

            t = n // hw
            x = x.reshape(b, t, h, w, c)

        if x.ndim != 5:
            return x

        b, t, hh, ww, c = x.shape
        kernel = self.laplacian_kernel.to(device=x.device, dtype=x.dtype)

        y = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, hh, ww)
        y = F.conv2d(y, kernel, padding=1, groups=c)
        y = y.reshape(b, t, c, hh, ww).permute(0, 1, 3, 4, 2)

        out = x + self.out(y)

        if init_ndim == 3:
            out = out.reshape(b, t * h * w, c)

        return out


class RuntimeSubjectCrossAttention(nn.Module):
    def __init__(self, query_dim, text_dim, num_heads=16):
        super().__init__()

        if query_dim % num_heads != 0:
            num_heads = 8 if query_dim % 8 == 0 else 1

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(text_dim, query_dim, bias=False)
        self.to_v = nn.Linear(text_dim, query_dim, bias=False)
        self.to_out = ZeroLinear(query_dim)

    def forward(self, x, text_embeds):
        init_ndim = x.ndim
        shape_5d = None

        if init_ndim == 5:
            b, t, h, w, c = x.shape
            shape_5d = (b, t, h, w, c)
            x = x.reshape(b, t * h * w, c)

        if x.ndim != 3 or text_embeds is None or text_embeds.ndim != 3:
            return x

        b, n, _ = x.shape

        if text_embeds.shape[0] != b:
            if text_embeds.shape[0] == 1:
                text_embeds = text_embeds.expand(b, -1, -1)
            else:
                return x

        m = text_embeds.shape[1]

        q = self.to_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(text_embeds).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(text_embeds).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, -1)
        out = x + self.to_out(out)

        if init_ndim == 5 and shape_5d is not None:
            out = out.reshape(*shape_5d)

        return out


class RuntimeStyleCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim, num_heads=16):
        super().__init__()

        if query_dim % num_heads != 0:
            num_heads = 8 if query_dim % 8 == 0 else 1

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.context_dim = context_dim

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = ZeroLinear(query_dim)

    def forward(self, x, style_context):
        init_ndim = x.ndim
        shape_5d = None

        if init_ndim == 5:
            b, t, h, w, c = x.shape
            shape_5d = (b, t, h, w, c)
            x = x.reshape(b, t * h * w, c)

        if x.ndim != 3 or style_context is None or style_context.ndim != 3:
            return torch.zeros_like(x) if init_ndim == 3 else torch.zeros(*shape_5d, device=x.device, dtype=x.dtype)

        b, n, _ = x.shape

        if style_context.shape[0] != b:
            if style_context.shape[0] == 1:
                style_context = style_context.expand(b, -1, -1)
            else:
                return x * 0.0

        style_context = _match_last_dim(style_context, self.context_dim)

        m = style_context.shape[1]

        q = self.to_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(style_context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(style_context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, -1)
        out = self.to_out(out)

        if init_ndim == 5 and shape_5d is not None:
            out = out.reshape(*shape_5d)

        return out


class RuntimeAnimaAdapter(nn.Module):
    def __init__(
            self,
            dit_hidden_size=2048,
            text_embed_dim=1024,
            style_dim=768,
            num_blocks=28,
            local_count=0,
            edge_count=0,
            subject_count=0,
            style_count=0,
            local_start=None,
            edge_start=0,
            subject_start=6,
            style_start=None,
            subject_heads=16,
            style_heads=16,
            has_semantic=False,
            has_mod_text=False,
            has_contrast=False,
            has_color=False,
    ):
        super().__init__()

        self.dit_hidden_size = dit_hidden_size
        self.text_embed_dim = text_embed_dim
        self.style_dim = style_dim
        self.num_blocks = num_blocks

        self.local_count = local_count
        self.edge_count = edge_count
        self.subject_count = subject_count
        self.style_count = style_count

        self.local_start = num_blocks - local_count if local_start is None else local_start
        self.edge_start = edge_start
        self.subject_start = subject_start
        self.style_start = num_blocks - style_count if style_start is None else style_start

        self.has_semantic = has_semantic
        self.has_mod_text = has_mod_text
        self.has_contrast = has_contrast
        self.has_color = has_color

        self.semantic_scale = RuntimeSemanticScaleAdapter(text_embed_dim) if has_semantic else None
        self.mod_text = RuntimeModTextAdapter(text_embed_dim, dit_hidden_size) if has_mod_text else None
        self.contrast_mod = RuntimeContrastModAdapter(text_embed_dim) if has_contrast else None
        self.color_tune = RuntimeColorTuneAdapter(text_embed_dim) if has_color else None

        self.local_convs = nn.ModuleList(
            [RuntimeLocalConvAdapter(dit_hidden_size) for _ in range(local_count)]
        )

        self.edge_details = nn.ModuleList(
            [RuntimeEdgeDetailConv(dit_hidden_size) for _ in range(edge_count)]
        )

        self.subject_blocks = nn.ModuleList(
            [
                RuntimeSubjectCrossAttention(
                    query_dim=dit_hidden_size,
                    text_dim=text_embed_dim,
                    num_heads=subject_heads,
                )
                for _ in range(subject_count)
            ]
        )

        self.style_blocks = nn.ModuleList(
            [
                RuntimeStyleCrossAttention(
                    query_dim=dit_hidden_size,
                    context_dim=style_dim,
                    num_heads=style_heads,
                )
                for _ in range(style_count)
            ]
        )

        if style_count > 0:
            self.style_scale = nn.Parameter(torch.zeros(style_count))
        else:
            self.register_parameter("style_scale", None)

        self._gamma = None
        self._beta = None
        self._text_cache = None
        self._warned_no_style = False
        self._warned_style_dim = False

    @staticmethod
    def _normalize_state_dict(sd):
        out = {}

        has_style_blocks = any(
            k.startswith("style_blocks.") or k.startswith("adapter.style_blocks.")
            for k in sd.keys()
        )

        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("adapter."):
                k = k[len("adapter."):]

            if k.startswith("__") or k.startswith("metadata"):
                continue

            if k == "style_scale" and not has_style_blocks:
                print("ℹ️ [AnimaAdapter] 权重里只有 style_scale 但没有 style_blocks，已忽略")
                continue

            out[k] = v

        return out

    @staticmethod
    def _count_indexed(sd, prefix):
        max_idx = -1
        for k in sd.keys():
            if not k.startswith(prefix + "."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1

    @classmethod
    def from_state_dict(cls, sd, num_blocks=28):
        sd = cls._normalize_state_dict(sd)

        dit_hidden_size = 2048
        text_embed_dim = 1024
        style_dim = 768

        if "semantic_scale.scale" in sd:
            text_embed_dim = sd["semantic_scale.scale"].numel()

        if "contrast_mod.gamma" in sd:
            text_embed_dim = sd["contrast_mod.gamma"].numel()

        if "color_tune.shift" in sd:
            text_embed_dim = sd["color_tune.shift"].numel()

        if "mod_text.text_proj.0.weight" in sd:
            w = sd["mod_text.text_proj.0.weight"]
            text_embed_dim = w.shape[1]
            dit_hidden_size = w.shape[0] // 2

        for k, v in sd.items():
            if k.endswith("dwconv.weight") and v.ndim == 4:
                dit_hidden_size = v.shape[0]
                break

        for k, v in sd.items():
            if k.endswith("to_q.weight") and v.ndim == 2:
                dit_hidden_size = v.shape[0]
                break

        for k, v in sd.items():
            if k.startswith("style_blocks.") and k.endswith("to_k.weight") and v.ndim == 2:
                style_dim = v.shape[1]
                break

        local_count = cls._count_indexed(sd, "local_convs")
        edge_count = cls._count_indexed(sd, "edge_details")
        subject_count = cls._count_indexed(sd, "subject_blocks")
        style_count = cls._count_indexed(sd, "style_blocks")

        has_semantic = any(k.startswith("semantic_scale.") for k in sd.keys())
        has_mod_text = any(k.startswith("mod_text.") for k in sd.keys())
        has_contrast = any(k.startswith("contrast_mod.") for k in sd.keys())
        has_color = any(k.startswith("color_tune.") for k in sd.keys())

        adapter = cls(
            dit_hidden_size=dit_hidden_size,
            text_embed_dim=text_embed_dim,
            style_dim=style_dim,
            num_blocks=num_blocks,
            local_count=local_count,
            edge_count=edge_count,
            subject_count=subject_count,
            style_count=style_count,
            local_start=None,
            edge_start=0,
            subject_start=6,
            style_start=None,
            has_semantic=has_semantic,
            has_mod_text=has_mod_text,
            has_contrast=has_contrast,
            has_color=has_color,
        )

        if style_count > 0 and "style_scale" in sd:
            old = sd["style_scale"]
            if old.numel() != style_count:
                fixed = torch.zeros(style_count, dtype=old.dtype, device=old.device)
                n = min(style_count, old.numel())
                fixed[:n] = old[:n]
                sd["style_scale"] = fixed

        missing, unexpected = adapter.load_state_dict(sd, strict=False)

        if missing:
            print(f"⚠️ [AnimaAdapter] 缺失权重: {len(missing)}")
            print(f"   前几个缺失: {missing[:5]}")

        if unexpected:
            print(f"⚠️ [AnimaAdapter] 额外权重: {len(unexpected)}")
            print(f"   前几个额外: {unexpected[:5]}")

        print(
            "ℹ️ [AnimaAdapter] 结构: "
            f"semantic={has_semantic}, mod_text={has_mod_text}, "
            f"contrast={has_contrast}, color={has_color}, "
            f"local={local_count}@{adapter.local_start}, "
            f"edge={edge_count}@{adapter.edge_start}, "
            f"subject={subject_count}@{adapter.subject_start}, "
            f"style={style_count}@{adapter.style_start}, style_dim={style_dim}"
        )

        return adapter

    def prepare_text(self, text, strength=1.0):
        if text is None or not torch.is_tensor(text) or text.ndim != 3:
            return text

        param = _module_first_param(self)
        if param is not None and (param.device != text.device or param.dtype != text.dtype):
            self.to(device=text.device, dtype=text.dtype)

        x = text

        if self.semantic_scale is not None:
            y = self.semantic_scale(x)
            x = x + strength * (y - x)

        if self.contrast_mod is not None:
            y = self.contrast_mod(x)
            x = x + strength * (y - x)

        if self.color_tune is not None:
            y = self.color_tune(x)
            x = x + strength * (y - x)

        if self.mod_text is not None:
            gamma, beta = self.mod_text(x)
            self._gamma = gamma.detach()
            self._beta = beta.detach()
        else:
            self._gamma = None
            self._beta = None

        self._text_cache = x.detach()
        return x

    def apply_mod_text(self, hidden, strength=1.0):
        if self._gamma is None or self._beta is None:
            return hidden

        gamma = self._gamma.to(device=hidden.device, dtype=hidden.dtype)
        beta = self._beta.to(device=hidden.device, dtype=hidden.dtype)

        if hidden.ndim == 3:
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]
        elif hidden.ndim == 5:
            gamma = gamma[:, None, None, None, :]
            beta = beta[:, None, None, None, :]
        else:
            return hidden

        if gamma.shape[0] != hidden.shape[0]:
            if gamma.shape[0] == 1:
                gamma = gamma.expand(hidden.shape[0], *gamma.shape[1:])
                beta = beta.expand(hidden.shape[0], *beta.shape[1:])
            else:
                return hidden

        y = hidden * (1.0 + gamma) + beta
        return hidden + strength * (y - hidden)

    def apply_block(self, block_idx, hidden, text_context=None, style_context=None, strength=1.0):
        param = _module_first_param(self)
        if param is not None and (param.device != hidden.device or param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

        hidden = self.apply_mod_text(hidden, strength=strength)

        h = None
        w = None

        if hidden.ndim == 5:
            h = hidden.shape[2]
            w = hidden.shape[3]
        elif hidden.ndim == 3:
            n = hidden.shape[1]
            side = int(math.sqrt(n))
            if side * side == n:
                h = side
                w = side

        if self.edge_count > 0 and self.edge_start <= block_idx < self.edge_start + self.edge_count:
            i = block_idx - self.edge_start
            if 0 <= i < len(self.edge_details):
                y = self.edge_details[i](hidden, h, w)
                hidden = hidden + strength * (y - hidden)

        if self.subject_count > 0 and self.subject_start <= block_idx < self.subject_start + self.subject_count:
            i = block_idx - self.subject_start
            if 0 <= i < len(self.subject_blocks):
                ctx = text_context if text_context is not None else self._text_cache
                if ctx is not None:
                    ctx = ctx.to(device=hidden.device, dtype=hidden.dtype)
                    y = self.subject_blocks[i](hidden, ctx)
                    hidden = hidden + strength * (y - hidden)

        if self.local_count > 0 and self.local_start <= block_idx < self.local_start + self.local_count:
            i = block_idx - self.local_start
            if 0 <= i < len(self.local_convs):
                y = self.local_convs[i](hidden, h, w)
                hidden = hidden + strength * (y - hidden)

        if self.style_count > 0 and self.style_start <= block_idx < self.style_start + self.style_count:
            i = block_idx - self.style_start
            if 0 <= i < len(self.style_blocks):
                if style_context is None:
                    if not self._warned_no_style:
                        print("⚠️ [AnimaAdapter] Adapter 含 StyleAttn，但没有传入 style_embeds，StyleAttn 本次不会生效")
                        self._warned_no_style = True
                else:
                    ctx = style_context.to(device=hidden.device, dtype=hidden.dtype)
                    ctx = _match_last_dim(ctx, self.style_dim)
                    style_out = self.style_blocks[i](hidden, ctx)
                    scale = self.style_scale[i].to(device=hidden.device, dtype=hidden.dtype)
                    hidden = hidden + strength * scale * style_out

        return hidden


def _is_text_tensor(x, text_dim):
    return torch.is_tensor(x) and x.ndim == 3 and x.shape[-1] == text_dim


def _prepare_context_in_call(args, kwargs, runtime_adapters):
    args = list(args)
    kwargs = dict(kwargs)
    text_context = None

    def apply_all(x):
        nonlocal text_context
        y = x
        for adapter, strength in runtime_adapters:
            y = adapter.prepare_text(y, strength=strength)
        text_context = y
        return y

    for key in list(kwargs.keys()):
        val = kwargs[key]
        if torch.is_tensor(val):
            for adapter, _ in runtime_adapters:
                if _is_text_tensor(val, adapter.text_embed_dim):
                    kwargs[key] = apply_all(val)
                    return tuple(args), kwargs, text_context

    for i, val in enumerate(args):
        if torch.is_tensor(val):
            for adapter, _ in runtime_adapters:
                if _is_text_tensor(val, adapter.text_embed_dim):
                    args[i] = apply_all(val)
                    return tuple(args), kwargs, text_context

    return tuple(args), kwargs, text_context


def _adapter_stack_signature(adapter_stack, device="auto", dtype="auto", num_blocks=28, style_embeds=None):
    signature = []

    for item in adapter_stack or []:
        name = item.get("name", "None")
        strength = float(item.get("strength", 1.0))

        if not name or name == "None" or abs(strength) < 1e-6:
            continue

        signature.append((name, strength))

    style_sig = None
    if torch.is_tensor(style_embeds):
        style_sig = (tuple(style_embeds.shape), str(style_embeds.dtype))

    return tuple(signature), str(device), str(dtype), int(num_blocks), style_sig


def _resolve_adapter_runtime_dtype(adapter_dtype, file_dtype, model_dtype):
    if adapter_dtype == "from_file":
        return file_dtype or model_dtype or torch.bfloat16

    if adapter_dtype == "auto":
        return model_dtype or file_dtype or torch.bfloat16

    return _dtype_from_string(adapter_dtype, model_dtype or file_dtype or torch.bfloat16)


def apply_anima_adapters_runtime(
        model_patcher,
        adapter_stack,
        device="auto",
        dtype="auto",
        num_blocks=28,
        style_embeds=None,
):
    diffusion_model = _find_diffusion_model_from_patcher(model_patcher)

    target_device = _device_from_string(device)
    model_dtype = _module_dtype(diffusion_model, torch.bfloat16)

    llm_dtype = model_dtype if dtype in ["auto", "from_file"] else _dtype_from_string(dtype, model_dtype)

    _force_anima_llm_adapter_device(diffusion_model, target_device, llm_dtype)
    _patch_anima_preprocess_text_embeds(diffusion_model)

    active_stack = [
        item for item in adapter_stack or []
        if item.get("name") and item.get("name") != "None" and abs(float(item.get("strength", 1.0))) > 1e-6
    ]

    if not active_stack:
        return model_patcher

    signature = _adapter_stack_signature(
        active_stack,
        device=device,
        dtype=dtype,
        num_blocks=num_blocks,
        style_embeds=style_embeds,
    )

    if getattr(model_patcher, "_anima_runtime_adapter_patched", False):
        old_signature = getattr(model_patcher, "_anima_runtime_adapter_signature", None)

        if old_signature == signature:
            print("ℹ️ [AnimaAdapter] Adapter 配置未变化，沿用已挂载插件")
            return model_patcher

        raise RuntimeError(
            "当前 MODEL 已经挂载过另一组 Anima Adapter。"
            "请重新执行模型烧录器节点，或重新加载工作流后再更换 Adapter / 强度 / dtype / style_embeds。"
        )

    runtime_adapters = []

    for item in active_stack:
        name = item.get("name", "None")
        strength = float(item.get("strength", 1.0))

        path = folder_paths.get_full_path(ANIMA_ADAPTER_FOLDER_NAME, name)
        if path is None:
            print(f"⚠️ [AnimaAdapter] 找不到插件: {name}")
            continue

        print(f"🧩 [AnimaAdapter] 加载插件: {path} | strength={strength}")

        sd = comfy.utils.load_torch_file(path, safe_load=True)

        file_dtype = _floating_dtype_from_state_dict(sd, None)
        runtime_dtype = _resolve_adapter_runtime_dtype(dtype, file_dtype, model_dtype)

        print(
            f"   ↳ 文件dtype: {_dtype_name(file_dtype)} | "
            f"底模dtype: {_dtype_name(model_dtype)} | "
            f"运行dtype: {_dtype_name(runtime_dtype)}"
        )

        adapter = RuntimeAnimaAdapter.from_state_dict(sd, num_blocks=num_blocks)
        adapter.to(device=target_device, dtype=runtime_dtype)
        adapter.eval()

        param_count = sum(p.numel() for p in adapter.parameters())
        bytes_per_param = torch.tensor([], dtype=runtime_dtype).element_size()
        size_mb = param_count * bytes_per_param / 1024 / 1024

        print(
            f"   ↳ 参数量: {param_count:,} | "
            f"约 {size_mb:.2f} MB ({runtime_dtype})"
        )

        runtime_adapters.append((adapter, strength))

        del sd
        gc.collect()

    if not runtime_adapters:
        print("⚠️ [AnimaAdapter] 没有有效插件")
        return model_patcher

    if not hasattr(diffusion_model, "blocks"):
        raise RuntimeError("Anima DiT 不存在 blocks，无法挂载插件")

    if torch.is_tensor(style_embeds):
        if style_embeds.ndim == 2:
            style_embeds = style_embeds.unsqueeze(1)
        elif style_embeds.ndim > 3:
            style_embeds = style_embeds.reshape(style_embeds.shape[0], -1, style_embeds.shape[-1])

        style_embeds = style_embeds.detach()
        print(f"🎨 [AnimaAdapter] 已接收 style_embeds: shape={tuple(style_embeds.shape)}, dtype={style_embeds.dtype}")
    else:
        style_embeds = None

    for idx, block in enumerate(diffusion_model.blocks):
        original_forward = block.forward

        def new_forward(block_self, *args, _idx=idx, _orig=original_forward, **kwargs):
            new_args, new_kwargs, text_context = _prepare_context_in_call(
                args,
                kwargs,
                runtime_adapters,
            )
            out = _orig(*new_args, **new_kwargs)

            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out

            if torch.is_tensor(hidden):
                for adapter, strength in runtime_adapters:
                    hidden = adapter.apply_block(
                        _idx,
                        hidden,
                        text_context=text_context,
                        style_context=style_embeds,
                        strength=strength,
                    )

            if is_tuple:
                return (hidden,) + tuple(out[1:])

            return hidden

        block.forward = types.MethodType(new_forward, block)

    model_patcher._anima_runtime_adapter_patched = True
    model_patcher._anima_runtime_adapter_signature = signature
    model_patcher._anima_runtime_adapters = runtime_adapters
    model_patcher._anima_runtime_style_embeds = style_embeds

    print(f"✅ [AnimaAdapter] 已挂载 {len(runtime_adapters)} 个 Adapter 插件")
    return model_patcher


class AnimaStyleEmbedsFromClipVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "target_dim": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ANIMA_STYLE_EMBEDS",)
    RETURN_NAMES = ("style_embeds",)
    FUNCTION = "make"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def make(self, clip_vision_output, target_dim=768, normalize=False):
        x = _extract_clip_vision_tensor(clip_vision_output)
        if x is None:
            raise RuntimeError("无法从 CLIP_VISION_OUTPUT 中提取 image_embeds / pooled_output / hidden_state")

        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        elif x.ndim != 3:
            raise RuntimeError(f"style_embeds 维度不支持: shape={tuple(x.shape)}")

        x = x.detach().float()

        if normalize:
            x = F.normalize(x, dim=-1)

        x = _match_last_dim(x, int(target_dim))

        print(f"🎨 [AnimaStyle] style_embeds: shape={tuple(x.shape)}, dtype={x.dtype}")
        return (x,)


class AnimaAdapterStack:
    @classmethod
    def INPUT_TYPES(cls):
        adapters = _adapter_list()
        return {
            "required": {
                "adapter_1": (adapters,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "adapter_2": (adapters,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "adapter_3": (adapters,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "adapter_4": (adapters,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "prev_stack": ("ANIMA_ADAPTER_STACK",),
            },
        }

    RETURN_TYPES = ("ANIMA_ADAPTER_STACK",)
    RETURN_NAMES = ("anima_adapter_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def build_stack(
            self,
            adapter_1,
            strength_1,
            adapter_2,
            strength_2,
            adapter_3,
            strength_3,
            adapter_4,
            strength_4,
            prev_stack=None,
    ):
        stack = list(prev_stack) if prev_stack is not None else []

        for name, strength in [
            (adapter_1, strength_1),
            (adapter_2, strength_2),
            (adapter_3, strength_3),
            (adapter_4, strength_4),
        ]:
            if name and name != "None" and abs(float(strength)) > 1e-6:
                stack.append({"name": name, "strength": float(strength)})

        return (stack,)


class AnimaDeviceFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "device": (["auto", "cpu", "cuda", "npu"], {"default": "auto"}),
                "dtype": (DTYPE_CHOICES,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "fix"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def fix(self, model, device="auto", dtype="auto"):
        diffusion_model = _find_diffusion_model_from_patcher(model)
        target_device = _device_from_string(device)

        model_dtype = _module_dtype(diffusion_model, torch.bfloat16)
        target_dtype = model_dtype if dtype in ["auto", "from_file"] else _dtype_from_string(dtype, model_dtype)

        _force_anima_llm_adapter_device(diffusion_model, target_device, target_dtype)
        _patch_anima_preprocess_text_embeds(diffusion_model)

        print(f"✅ [AnimaDeviceFix] llm_adapter -> {target_device}, {target_dtype}")
        return (model,)


class SeparateModelMixerDictFuser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"),),
                "clip": (folder_paths.get_filename_list("clip"),),
                "vae": (folder_paths.get_filename_list("vae"),),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "anima_adapter_stack": ("ANIMA_ADAPTER_STACK",),
                "style_embeds": ("ANIMA_STYLE_EMBEDS",),
                "save_dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
                "device": (["auto", "cpu", "cuda", "npu"], {"default": "auto"}),
                "adapter_device": (["auto", "cpu", "cuda", "npu"], {"default": "auto"}),
                "adapter_dtype": (DTYPE_CHOICES, {"default": "auto"}),
                "adapter_num_blocks": ("INT", {"default": 28, "min": 1, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "pure_dict_merge"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def pure_dict_merge(
            self,
            model,
            clip,
            vae,
            lora_stack=None,
            anima_adapter_stack=None,
            style_embeds=None,
            save_dtype="auto",
            device="auto",
            adapter_device="auto",
            adapter_dtype="auto",
            adapter_num_blocks=28,
    ):
        lora_stack = [] if lora_stack is None else lora_stack
        anima_adapter_stack = [] if anima_adapter_stack is None else anima_adapter_stack

        active_loras = [
            l for l in lora_stack
            if l[0] and l[0] != "None" and (abs(l[1]) > 0.0001 or abs(l[2]) > 0.0001)
        ]

        active_adapters = [
            a for a in anima_adapter_stack
            if a.get("name") and a.get("name") != "None" and abs(float(a.get("strength", 1.0))) > 1e-6
        ]

        print("🧹 [TrueMixer] 清空残留显存...")
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()

        target_device = _device_from_string(device)
        print(f"🔧 [TrueMixer] 使用设备: {target_device}")

        def find_model_path(model_name, folder_types):
            for folder_type in folder_types:
                path = folder_paths.get_full_path(folder_type, model_name)
                if path is not None:
                    return path, folder_type
            return None, None

        model_path, _ = find_model_path(model, ["checkpoints"])
        if model_path is None:
            raise RuntimeError(f"找不到主模型文件: {model}")

        vae_path, _ = find_model_path(vae, ["vae"])
        if vae_path is None:
            raise RuntimeError(f"找不到 VAE 模型文件: {vae}")

        clip_path, _ = find_model_path(clip, ["clip"])
        if clip_path is None:
            raise RuntimeError(f"找不到 CLIP 模型文件: {clip}")

        print(f"📥 [TrueMixer] 加载 Anima 基础模型: {model_path}")
        ckpt_out = comfy.sd.load_checkpoint_guess_config(
            model_path,
            output_vae=False,
            output_clip=False,
        )
        model_obj = ckpt_out[0]

        print(f"📥 [TrueMixer] 加载 Anima VAE: {vae_path}")
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae_obj = comfy.sd.VAE(sd=vae_sd)
        del vae_sd
        gc.collect()

        base_dtype = next(model_obj.model.parameters()).dtype
        target_dtype = _dtype_from_string(save_dtype, base_dtype)

        print(f"🔧 [TrueMixer] 底模文件dtype: {base_dtype} | 烧录保存dtype: {target_dtype}")

        print(f"📥 [TrueMixer] 加载 Anima Qwen3 CLIP: {clip_path}")

        if AnimaTEModel is None or AnimaTokenizer is None:
            raise RuntimeError(
                "当前 ComfyUI 环境中找不到 comfy.text_encoders.anima.AnimaTEModel / AnimaTokenizer"
            )

        clip_sd = comfy.utils.load_torch_file(clip_path, safe_load=True)

        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=AnimaTokenizer,
            clip=AnimaTEModel,
        )

        clip_obj = CLIP(clip_target, embedding_directory=None)
        clip_obj.load_sd(clip_sd)

        del clip_sd
        gc.collect()

        model_clone = model_obj.clone()

        if hasattr(clip_obj, "clone"):
            clip_clone = clip_obj.clone()
        else:
            clip_clone = copy.copy(clip_obj)

        print(f"🛠️ [TrueMixer] LoRA 数量: {len(active_loras)}")

        for lora_name, m_strength, c_strength in active_loras:
            lora_path, _ = find_model_path(lora_name, ["loras"])
            if lora_path is None:
                print(f"⚠️ [TrueMixer] 找不到 LoRA: {lora_name}")
                continue

            print(f"  - 应用 LoRA: {lora_name} | model={m_strength}, clip={c_strength}")
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

            if hasattr(clip_clone, "cond_stage_model"):
                model_clone, clip_clone = comfy.sd.load_lora_for_models(
                    model_clone,
                    clip_clone,
                    lora_sd,
                    m_strength,
                    c_strength,
                )
            else:
                model_clone, _ = comfy.sd.load_lora_for_models(
                    model_clone,
                    None,
                    lora_sd,
                    m_strength,
                    c_strength,
                )

            del lora_sd
            gc.collect()

        @torch.inference_mode()
        def bake_model_weights(patcher, name="MODEL", force_cpu=False):
            calc_device = torch.device("cpu") if force_cpu else target_device

            print(f"🔥 [TrueMixer] 使用 [{calc_device}] 以 {target_dtype} 烧录 {name}...")

            model_inner = patcher.model
            model_inner.to(calc_device)

            sd = model_inner.state_dict()
            new_sd = {}
            patches = getattr(patcher, "patches", {})

            total = len(sd)

            for i, key in enumerate(sd.keys()):
                if i % 100 == 0:
                    print(f"  - {name}: {i + 1}/{total}")

                weight = sd[key].to(calc_device).to(torch.float32)

                if key in patches:
                    weight = comfy.lora.calculate_weight(patches[key], weight, key)

                if target_dtype == torch.float16:
                    weight = torch.nan_to_num(
                        weight,
                        nan=0.0,
                        posinf=65504,
                        neginf=-65504,
                    )

                new_sd[key] = weight.to(target_dtype).cpu()

            patcher.patches = {}
            patcher.backup = {}

            if hasattr(patcher, "object_patches"):
                patcher.object_patches = {}

            model_inner.load_state_dict(new_sd, strict=False)

            if hasattr(patcher, "base_model"):
                try:
                    patcher.base_model.load_state_dict(new_sd, strict=False)
                except Exception:
                    pass

            if force_cpu:
                model_inner.to(torch.device("cpu"))
            else:
                model_inner.to(target_device)

            return patcher

        if active_loras or target_dtype != base_dtype:
            model_clone = bake_model_weights(
                model_clone,
                "UNET/Transformer",
                force_cpu=(device == "cpu"),
            )

            if hasattr(clip_clone, "patcher") and clip_clone.patcher is not None:
                print("🔥 [TrueMixer] 正在烧录 CLIP...")
                clip_clone.patcher = bake_model_weights(
                    clip_clone.patcher,
                    "CLIP",
                    force_cpu=(device == "cpu"),
                )
            else:
                print("⚠️ [TrueMixer] CLIP 没有 patcher，跳过 CLIP 烧录")

        diffusion_model = _find_diffusion_model_from_patcher(model_clone)

        runtime_device = _device_from_string(adapter_device)
        diffusion_dtype = _module_dtype(diffusion_model, target_dtype)

        if adapter_dtype in ["auto", "from_file"]:
            runtime_llm_dtype = diffusion_dtype
        else:
            runtime_llm_dtype = _dtype_from_string(adapter_dtype, diffusion_dtype)

        _force_anima_llm_adapter_device(diffusion_model, runtime_device, runtime_llm_dtype)
        _patch_anima_preprocess_text_embeds(diffusion_model)

        if active_adapters:
            print(f"🧩 [TrueMixer] 挂载 Anima Adapter 插件: {len(active_adapters)} 个")
            print(
                f"   ↳ adapter_dtype={adapter_dtype} "
                f"(auto=跟随底模, from_file=跟随插件文件)"
            )

            model_clone = apply_anima_adapters_runtime(
                model_clone,
                active_adapters,
                device=adapter_device,
                dtype=adapter_dtype,
                num_blocks=int(adapter_num_blocks),
                style_embeds=style_embeds,
            )

        gc.collect()
        comfy.model_management.soft_empty_cache()

        print("✅ [TrueMixer] 处理完成")
        return (model_clone, clip_clone, vae_obj)


NODE_CLASS_MAPPINGS = {
    "AnimaAdapterStack": AnimaAdapterStack,
    "AnimaDeviceFix": AnimaDeviceFix,
    "AnimaStyleEmbedsFromClipVision": AnimaStyleEmbedsFromClipVision,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterStack": "Anima Adapter 插件堆",
    "AnimaDeviceFix": "Anima 设备修复器",
    "AnimaStyleEmbedsFromClipVision": "Anima Style Embeds From CLIP Vision",
    "SeparateModelMixerDictFuser": "Anima模型烧录器",
}
