# coding=utf-8
# AnimaLoader.py
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


def _tensor_basic_stats(x):
    if not torch.is_tensor(x):
        return {
            "is_tensor": False,
            "shape": None,
            "dtype": None,
            "device": None,
            "finite": False,
            "all_zero": True,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "absmax": 0.0,
        }

    try:
        with torch.no_grad():
            xf = x.detach().float()
            finite = bool(torch.isfinite(xf).all().detach().cpu())

            if xf.numel() == 0:
                return {
                    "is_tensor": True,
                    "shape": tuple(x.shape),
                    "dtype": str(x.dtype),
                    "device": str(x.device),
                    "finite": finite,
                    "all_zero": True,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "absmax": 0.0,
                }

            mean_val = float(xf.mean().detach().cpu())
            std_val = float(xf.std().detach().cpu()) if xf.numel() > 1 else 0.0
            min_val = float(xf.min().detach().cpu())
            max_val = float(xf.max().detach().cpu())
            absmax_val = float(xf.abs().max().detach().cpu())

            return {
                "is_tensor": True,
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "finite": finite,
                "all_zero": absmax_val == 0.0,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "absmax": absmax_val,
            }
    except Exception:
        return {
            "is_tensor": True,
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "finite": False,
            "all_zero": False,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "absmax": 0.0,
        }


def _safe_tensor_stats_str(x, name="tensor"):
    if not torch.is_tensor(x):
        return f"{name}: not tensor"

    try:
        with torch.no_grad():
            xf = x.detach().float()
            if xf.numel() == 0:
                return f"{name}: shape={tuple(x.shape)}, empty"

            mean_val = float(xf.mean().detach().cpu())
            std_val = float(xf.std().detach().cpu()) if xf.numel() > 1 else 0.0
            min_val = float(xf.min().detach().cpu())
            max_val = float(xf.max().detach().cpu())
            absmax_val = float(xf.abs().max().detach().cpu())
            meanabs_val = float(xf.abs().mean().detach().cpu())

            return (
                f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
                f"mean={mean_val:.8f}, std={std_val:.8f}, "
                f"min={min_val:.8f}, max={max_val:.8f}, "
                f"absmax={absmax_val:.8f}, meanabs={meanabs_val:.8f}"
            )
    except Exception as e:
        return f"{name}: shape={tuple(x.shape)}, stats_error={e}"


def _module_param_absmax(module):
    if module is None:
        return 0.0

    max_val = 0.0
    try:
        with torch.no_grad():
            for p in module.parameters():
                if p is None:
                    continue
                if p.numel() == 0:
                    continue
                v = float(p.detach().float().abs().max().cpu())
                max_val = max(max_val, v)
    except Exception:
        pass

    return max_val


def _collect_clip_vision_tensors(clip_vision_output):
    """
    收集 CLIP_VISION_OUTPUT 里可能存在的 tensor。

    兼容：
    1. tensor
    2. dict
    3. 带属性的对象
    4. tuple/list
    """
    candidates = []

    if clip_vision_output is None:
        return candidates

    if torch.is_tensor(clip_vision_output):
        candidates.append(("tensor", clip_vision_output))
        return candidates

    preferred_keys = [
        "image_embeds",
        "pooled_output",
        "last_hidden_state",
        "penultimate_hidden_states",
        "cond",
    ]

    if isinstance(clip_vision_output, dict):
        for key in preferred_keys:
            v = clip_vision_output.get(key, None)
            if torch.is_tensor(v):
                candidates.append((key, v))

        for key, v in clip_vision_output.items():
            if torch.is_tensor(v) and all(key != k for k, _ in candidates):
                candidates.append((str(key), v))

        return candidates

    for key in preferred_keys:
        if hasattr(clip_vision_output, key):
            try:
                v = getattr(clip_vision_output, key)
                if torch.is_tensor(v):
                    candidates.append((key, v))
            except Exception:
                pass

    if isinstance(clip_vision_output, (list, tuple)):
        for i, v in enumerate(clip_vision_output):
            if torch.is_tensor(v):
                candidates.append((f"item_{i}", v))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        candidates.append((f"item_{i}.{kk}", vv))

    return candidates


def _extract_clip_vision_tensor(
        clip_vision_output,
        preferred_key=None,
        allow_zero=False,
        verbose=False,
):
    """
    从 CLIP_VISION_OUTPUT 中提取有效 tensor。

    修复点：
    1. 不再盲目优先返回 image_embeds。
    2. 如果 image_embeds 是全 0，会继续查找 pooled_output / hidden_state 等非零 tensor。
    3. 支持 preferred_key 指定字段。
    4. 支持 allow_zero=True 强制允许返回全 0 tensor。
    """
    candidates = _collect_clip_vision_tensors(clip_vision_output)

    if not candidates:
        return None

    priority = {
        "image_embeds": 0,
        "pooled_output": 1,
        "penultimate_hidden_states": 2,
        "last_hidden_state": 3,
        "cond": 4,
        "tensor": 5,
    }

    inspected = []

    for name, tensor in candidates:
        st = _tensor_basic_stats(tensor)
        inspected.append((name, tensor, st))

    if verbose:
        print("[AnimaStyle] CLIP Vision tensor candidates:")
        for name, tensor, st in inspected:
            print(
                f"  - {name}: "
                f"shape={st['shape']}, dtype={st['dtype']}, device={st['device']}, "
                f"finite={st['finite']}, all_zero={st['all_zero']}, "
                f"mean={st['mean']:.6f}, std={st['std']:.6f}, "
                f"min={st['min']:.6f}, max={st['max']:.6f}, absmax={st['absmax']:.6f}"
            )

    if preferred_key is not None and preferred_key != "auto_nonzero":
        for name, tensor, st in inspected:
            if name == preferred_key:
                if allow_zero or not st["all_zero"]:
                    return tensor
                print(
                    f"[AnimaStyle] 警告：指定字段 {preferred_key} 是全 0，"
                    f"将尝试寻找其它非零 CLIP Vision tensor。"
                )
                break

    valid = []
    zero_valid = []

    for name, tensor, st in inspected:
        if not st["finite"]:
            continue

        item = (
            priority.get(name, 100),
            0 if tensor.ndim == 2 else 1 if tensor.ndim == 3 else 2,
            name,
            tensor,
            st,
        )

        if st["all_zero"]:
            zero_valid.append(item)
        else:
            valid.append(item)

    if valid:
        valid.sort(key=lambda x: (x[0], x[1]))
        chosen = valid[0]

        if verbose:
            print(
                f"[AnimaStyle] 选择 CLIP Vision tensor: {chosen[2]} | "
                f"shape={tuple(chosen[3].shape)} | std={chosen[4]['std']:.6f}"
            )

        return chosen[3]

    if allow_zero and zero_valid:
        zero_valid.sort(key=lambda x: (x[0], x[1]))
        chosen = zero_valid[0]

        print(
            f"[AnimaStyle] 警告：所有 CLIP Vision tensor 都是全 0，"
            f"只能返回 {chosen[2]}。请检查 CLIP Vision Encode 节点和输入图片。"
        )

        return chosen[3]

    print(
        "[AnimaStyle] 错误：找到了 CLIP Vision tensor，"
        "但它们不是非有限值就是全 0。请使用 AnimaClipVisionInspector 节点检查。"
    )

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
    """
    Runtime LocalConv Adapter.

    修复点：
    1. 支持从 checkpoint 自动匹配 kernel_size，例如 3x3 / 5x5。
    2. dwconv.bias 强制零初始化，避免 checkpoint 没有 bias 时保留随机 bias 导致画风崩坏。
    3. 对偶数 kernel 或异常输出尺寸做安全裁剪/补边，避免 hidden 尺寸被卷积改坏。
    """

    def __init__(self, dim, kernel_size=3, use_bias=True):
        super().__init__()

        kernel_size = int(kernel_size)
        if kernel_size < 1:
            kernel_size = 3

        self.dim = int(dim)
        self.kernel_size = kernel_size

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=bool(use_bias),
        )

        # 关键修复：
        # Conv2d 默认 bias 是随机初始化。
        # 如果 checkpoint 里没有 dwconv.bias，load_state_dict(strict=False) 不会覆盖它。
        # 所以这里必须清零，避免画面偏移/风格崩坏。
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)

        self.pwconv = ZeroLinear(dim)

    @staticmethod
    def _match_hw(y, target_h, target_w):
        """
        卷积后空间尺寸保护。
        正常奇数 kernel + padding=kernel//2 时尺寸不变。
        如果是偶数 kernel 或特殊情况导致尺寸变化，则中心裁剪或补零回原尺寸。
        """
        if y.ndim != 4:
            return y

        _, _, h, w = y.shape

        # crop height
        if h > target_h:
            start = (h - target_h) // 2
            y = y[:, :, start:start + target_h, :]
        elif h < target_h:
            pad_total = target_h - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            y = F.pad(y, (0, 0, pad_top, pad_bottom))

        # crop width
        if w > target_w:
            start = (w - target_w) // 2
            y = y[:, :, :, start:start + target_w]
        elif w < target_w:
            pad_total = target_w - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            y = F.pad(y, (pad_left, pad_right, 0, 0))

        return y

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

        if c != self.dim:
            return x

        y = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, hh, ww)
        y = self.dwconv(y)

        # 安全防护：确保卷积不改变空间尺寸
        y = self._match_hw(y, hh, ww)

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
    """
    对齐 train_anima_adapter.py 里的 StyleCrossAttention。

    训练时 StyleCrossAttention.forward 返回的是 style_out，
    外层再执行：
        hidden = hidden + style_scale[i] * style_out

    所以这里也不在内部加残差。
    """

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
            local_kernel_size=3,
    ):
        super().__init__()

        self.dit_hidden_size = int(dit_hidden_size)
        self.text_embed_dim = int(text_embed_dim)
        self.style_dim = int(style_dim)
        self.num_blocks = int(num_blocks)

        self.local_count = int(local_count)
        self.edge_count = int(edge_count)
        self.subject_count = int(subject_count)
        self.style_count = int(style_count)

        self.local_start = self.num_blocks - self.local_count if local_start is None else int(local_start)
        self.edge_start = int(edge_start)
        self.subject_start = int(subject_start)
        self.style_start = self.num_blocks - self.style_count if style_start is None else int(style_start)

        self.has_semantic = bool(has_semantic)
        self.has_mod_text = bool(has_mod_text)
        self.has_contrast = bool(has_contrast)
        self.has_color = bool(has_color)

        if isinstance(local_kernel_size, (list, tuple)):
            local_kernel_sizes = [int(x) for x in local_kernel_size]
            if len(local_kernel_sizes) < self.local_count:
                local_kernel_sizes += [local_kernel_sizes[-1] if local_kernel_sizes else 3] * (
                        self.local_count - len(local_kernel_sizes)
                )
            local_kernel_sizes = local_kernel_sizes[:self.local_count]
        else:
            local_kernel_sizes = [int(local_kernel_size)] * self.local_count

        local_kernel_sizes = [k if k >= 1 else 3 for k in local_kernel_sizes]
        self.local_kernel_sizes = local_kernel_sizes

        self.semantic_scale = RuntimeSemanticScaleAdapter(self.text_embed_dim) if self.has_semantic else None
        self.mod_text = RuntimeModTextAdapter(self.text_embed_dim, self.dit_hidden_size) if self.has_mod_text else None
        self.contrast_mod = RuntimeContrastModAdapter(self.text_embed_dim) if self.has_contrast else None
        self.color_tune = RuntimeColorTuneAdapter(self.text_embed_dim) if self.has_color else None

        self.local_convs = nn.ModuleList(
            [
                RuntimeLocalConvAdapter(
                    self.dit_hidden_size,
                    kernel_size=local_kernel_sizes[i],
                    use_bias=True,
                )
                for i in range(self.local_count)
            ]
        )

        self.edge_details = nn.ModuleList(
            [RuntimeEdgeDetailConv(self.dit_hidden_size) for _ in range(self.edge_count)]
        )

        self.subject_blocks = nn.ModuleList(
            [
                RuntimeSubjectCrossAttention(
                    query_dim=self.dit_hidden_size,
                    text_dim=self.text_embed_dim,
                    num_heads=subject_heads,
                )
                for _ in range(self.subject_count)
            ]
        )

        self.style_blocks = nn.ModuleList(
            [
                RuntimeStyleCrossAttention(
                    query_dim=self.dit_hidden_size,
                    context_dim=self.style_dim,
                    num_heads=style_heads,
                )
                for _ in range(self.style_count)
            ]
        )

        # 关键修复：
        # 之前这里是 zeros，如果 checkpoint 没有 style_scale，Style 分支会被 scale=0 完全关闭。
        # 改为 ones，保证旧权重没有 style_scale 时 Style 分支仍然可以生效。
        if self.style_count > 0:
            self.style_scale = nn.Parameter(torch.ones(self.style_count))
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
                print("[AnimaAdapter] 权重里只有 style_scale 但没有 style_blocks，已忽略 style_scale")
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

    @staticmethod
    def _detect_local_kernel_sizes(sd, local_count):
        kernel_sizes = [3] * int(local_count)

        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue

            parts = k.split(".")

            if len(parts) != 4:
                continue

            if parts[0] != "local_convs":
                continue

            if not parts[1].isdigit():
                continue

            if parts[2] != "dwconv" or parts[3] != "weight":
                continue

            if v.ndim != 4:
                continue

            idx = int(parts[1])
            if idx < 0 or idx >= local_count:
                continue

            kh = int(v.shape[2])
            kw = int(v.shape[3])

            if kh != kw:
                print(
                    f"[AnimaAdapter] local_convs.{idx}.dwconv.weight 是非正方形卷积核: "
                    f"{kh}x{kw}，将使用 kh={kh}"
                )

            kernel_sizes[idx] = kh

        return kernel_sizes

    @staticmethod
    def _filter_state_dict_by_shape(model, sd):
        model_sd = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in sd.items():
            if k not in model_sd:
                filtered[k] = v
                continue

            if not torch.is_tensor(v) or not torch.is_tensor(model_sd[k]):
                filtered[k] = v
                continue

            if tuple(v.shape) != tuple(model_sd[k].shape):
                skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue

            filtered[k] = v

        if skipped:
            print(f"[AnimaAdapter] 跳过 shape 不匹配权重: {len(skipped)} 个")
            for item in skipped[:16]:
                print(f"   - {item[0]}: checkpoint={item[1]} runtime={item[2]}")
            if len(skipped) > 16:
                print(f"   ... 还有 {len(skipped) - 16} 个未显示")

        return filtered

    @classmethod
    def from_state_dict(cls, sd, num_blocks=28):
        sd = cls._normalize_state_dict(sd)

        dit_hidden_size = 2048
        text_embed_dim = 1024
        style_dim = 768

        if "semantic_scale.scale" in sd and torch.is_tensor(sd["semantic_scale.scale"]):
            text_embed_dim = sd["semantic_scale.scale"].numel()

        if "contrast_mod.gamma" in sd and torch.is_tensor(sd["contrast_mod.gamma"]):
            text_embed_dim = sd["contrast_mod.gamma"].numel()

        if "color_tune.shift" in sd and torch.is_tensor(sd["color_tune.shift"]):
            text_embed_dim = sd["color_tune.shift"].numel()

        if "mod_text.text_proj.0.weight" in sd and torch.is_tensor(sd["mod_text.text_proj.0.weight"]):
            w = sd["mod_text.text_proj.0.weight"]
            if w.ndim == 2:
                text_embed_dim = int(w.shape[1])
                dit_hidden_size = int(w.shape[0] // 2)

        for k, v in sd.items():
            if (
                    k.startswith("local_convs.")
                    and k.endswith(".dwconv.weight")
                    and torch.is_tensor(v)
                    and v.ndim == 4
            ):
                dit_hidden_size = int(v.shape[0])
                break

        for k, v in sd.items():
            if k.endswith("to_q.weight") and torch.is_tensor(v) and v.ndim == 2:
                dit_hidden_size = int(v.shape[0])
                break

        for k, v in sd.items():
            if (
                    k.startswith("style_blocks.")
                    and k.endswith("to_k.weight")
                    and torch.is_tensor(v)
                    and v.ndim == 2
            ):
                style_dim = int(v.shape[1])
                break

        local_count = cls._count_indexed(sd, "local_convs")
        edge_count = cls._count_indexed(sd, "edge_details")
        subject_count = cls._count_indexed(sd, "subject_blocks")
        style_count = cls._count_indexed(sd, "style_blocks")

        local_kernel_sizes = cls._detect_local_kernel_sizes(sd, local_count)

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
            local_kernel_size=local_kernel_sizes,
        )

        has_style_scale_in_file = "style_scale" in sd and torch.is_tensor(sd["style_scale"])

        if style_count > 0:
            if has_style_scale_in_file:
                old = sd["style_scale"]
                if old.numel() != style_count:
                    fixed = torch.ones(style_count, dtype=old.dtype, device=old.device)
                    n = min(style_count, old.numel())
                    fixed[:n] = old[:n]
                    sd["style_scale"] = fixed
                    print(
                        f"[AnimaAdapter] style_scale 数量不匹配，已修正: "
                        f"checkpoint={old.numel()} runtime={style_count}"
                    )
            else:
                print(
                    "[AnimaAdapter] 权重中没有 style_scale。"
                    "Runtime 将使用默认 style_scale=1.0。"
                )

        sd_to_load = cls._filter_state_dict_by_shape(adapter, sd)
        missing, unexpected = adapter.load_state_dict(sd_to_load, strict=False)

        missing_local_bias = [
            k for k in missing
            if k.startswith("local_convs.") and k.endswith(".dwconv.bias")
        ]

        missing_style_scale = [
            k for k in missing
            if k == "style_scale"
        ]

        if missing:
            print(f"[AnimaAdapter] 缺失权重: {len(missing)}")
            print(f"   前几个缺失: {missing[:8]}")

            if missing_local_bias:
                print(
                    f"[AnimaAdapter] 检测到 {len(missing_local_bias)} 个 local_conv dwconv.bias 缺失，"
                    f"已使用零初始化。"
                )

            if missing_style_scale:
                print("[AnimaAdapter] style_scale 缺失，已使用默认值 1.0。")

        if unexpected:
            print(f"[AnimaAdapter] 额外权重: {len(unexpected)}")
            print(f"   前几个额外: {unexpected[:8]}")

        adapter._style_to_k_absmax = 0.0
        adapter._style_to_v_absmax = 0.0
        adapter._style_to_out_absmax = 0.0

        if style_count > 0:
            if adapter.style_scale is not None:
                print("[AnimaAdapter] " + _safe_tensor_stats_str(adapter.style_scale, "style_scale"))

                try:
                    with torch.no_grad():
                        scale_absmax = float(adapter.style_scale.detach().float().abs().max().cpu())
                        scale_meanabs = float(adapter.style_scale.detach().float().abs().mean().cpu())

                    if scale_absmax < 1e-8:
                        print(
                            "[AnimaAdapter] ⚠️ 检测到 style_scale 全 0。"
                            "这会导致 StyleAttn 完全无效。Runtime 已临时将 style_scale 改为 1.0。"
                        )
                        with torch.no_grad():
                            adapter.style_scale.fill_(1.0)
                    elif scale_meanabs < 1e-4:
                        print(
                            "[AnimaAdapter] ⚠️ style_scale 数值极小，StyleAttn 影响可能非常弱。"
                        )
                except Exception as e:
                    print(f"[AnimaAdapter] style_scale 检查失败: {e}")

            style_to_k_absmax = 0.0
            style_to_v_absmax = 0.0
            style_to_out_absmax = 0.0

            for blk in adapter.style_blocks:
                style_to_k_absmax = max(style_to_k_absmax, _module_param_absmax(blk.to_k))
                style_to_v_absmax = max(style_to_v_absmax, _module_param_absmax(blk.to_v))
                style_to_out_absmax = max(style_to_out_absmax, _module_param_absmax(blk.to_out))

            adapter._style_to_k_absmax = float(style_to_k_absmax)
            adapter._style_to_v_absmax = float(style_to_v_absmax)
            adapter._style_to_out_absmax = float(style_to_out_absmax)

            print(
                f"[AnimaAdapter] Style 权重检查: "
                f"to_k_absmax={style_to_k_absmax:.8f}, "
                f"to_v_absmax={style_to_v_absmax:.8f}, "
                f"to_out_absmax={style_to_out_absmax:.8f}"
            )

            if style_to_k_absmax < 1e-8 or style_to_v_absmax < 1e-8:
                print(
                    "[AnimaAdapter] ⚠️ StyleAttn 的 to_k/to_v 权重接近全 0，"
                    "style_embeds 很可能无法被有效读取。"
                )

            if style_to_out_absmax < 1e-8:
                print(
                    "[AnimaAdapter] ⚠️ StyleAttn 的 to_out 权重接近全 0。"
                    "style_out 最终会被 to_out 压成接近 0，画风迁移会几乎无效。"
                )
            elif style_to_out_absmax < 1e-4:
                print(
                    "[AnimaAdapter] ⚠️ StyleAttn 的 to_out 权重非常小。"
                    f"当前 to_out_absmax={style_to_out_absmax:.8f}。"
                    "这通常会导致画风迁移很弱。"
                    "如果这是 epoch_1 权重，建议继续训练更多 epoch，"
                    "或者临时提高 Adapter strength 做验证。"
                )

        print(
            "[AnimaAdapter] 结构: "
            f"semantic={has_semantic}, mod_text={has_mod_text}, "
            f"contrast={has_contrast}, color={has_color}, "
            f"local={local_count}@{adapter.local_start}, "
            f"local_kernel={local_kernel_sizes}, "
            f"edge={edge_count}@{adapter.edge_start}, "
            f"subject={subject_count}@{adapter.subject_start}, "
            f"style={style_count}@{adapter.style_start}, "
            f"style_dim={style_dim}, "
            f"style_scale={'from_file' if has_style_scale_in_file else 'default_ones'}"
        )

        adapter._configure_style_runtime_boost(
            enable=True,
            target_to_out_absmax=2e-3,
            max_gain=128.0,
            min_problem_absmax=1e-4,
        )

        return adapter

    def prepare_text(self, text, strength=1.0):
        if text is None or not torch.is_tensor(text) or text.ndim != 3:
            return text

        param = _module_first_param(self)
        if param is not None and (param.device != text.device or param.dtype != text.dtype):
            self.to(device=text.device, dtype=text.dtype)

        x = text

        # 不额外削弱 contrast/color/semantic。
        # 如果训练时就是完整强度，这里也应该完整还原。
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
                    if not getattr(self, "_warned_no_style", False):
                        print(
                            "[AnimaAdapter] ⚠️ 当前 Adapter 包含 StyleAttn，"
                            "但没有传入 style_embeds，所以 StyleAttn 不会生效。"
                        )
                        self._warned_no_style = True
                else:
                    ctx = style_context

                    if not torch.is_tensor(ctx):
                        if not getattr(self, "_warned_no_style", False):
                            print(
                                "[AnimaAdapter] ⚠️ style_context 不是 tensor，"
                                "StyleAttn 不会生效。"
                            )
                            self._warned_no_style = True
                    else:
                        if ctx.ndim == 2:
                            ctx = ctx.unsqueeze(1)
                        elif ctx.ndim > 3:
                            ctx = ctx.reshape(ctx.shape[0], -1, ctx.shape[-1])
                        elif ctx.ndim != 3:
                            if not getattr(self, "_warned_no_style", False):
                                print(
                                    f"[AnimaAdapter] ⚠️ style_context 维度不支持: "
                                    f"shape={tuple(ctx.shape)}，StyleAttn 不会生效。"
                                )
                                self._warned_no_style = True
                            ctx = None

                        if ctx is not None:
                            source_dim = int(ctx.shape[-1])

                            if source_dim != self.style_dim and not getattr(self, "_warned_style_dim", False):
                                print(
                                    f"[AnimaAdapter] ⚠️ style_embeds 维度与 Adapter 不匹配: "
                                    f"style_embeds_dim={source_dim}, adapter_style_dim={self.style_dim}。\n"
                                    f"代码会临时执行裁剪/补零以避免报错，但这不是正确的风格投影，"
                                    f"画风迁移可能很弱或近似无效。"
                                )
                                self._warned_style_dim = True

                            ctx = ctx.to(device=hidden.device, dtype=hidden.dtype)
                            ctx = _match_last_dim(ctx, self.style_dim)

                            style_out = self.style_blocks[i](hidden, ctx)

                            scale = self.style_scale[i].to(device=hidden.device, dtype=hidden.dtype)

                            runtime_gain = float(getattr(self, "_style_runtime_gain", 1.0))
                            style_delta = strength * scale * runtime_gain * style_out

                            hidden = hidden + style_delta

                            if not getattr(self, "_style_active_printed", False):
                                self._style_active_printed = True

                                try:
                                    ctx_stats = _safe_tensor_stats_str(ctx, "style_ctx")
                                    hidden_stats = _safe_tensor_stats_str(hidden, "hidden_after_style")
                                    out_stats = _safe_tensor_stats_str(style_out, "style_out")
                                    delta_stats = _safe_tensor_stats_str(style_delta, "style_delta")

                                    scale_val = float(scale.detach().float().cpu())
                                    hidden_meanabs = float(hidden.detach().float().abs().mean().cpu())
                                    delta_meanabs = float(style_delta.detach().float().abs().mean().cpu())

                                    if hidden_meanabs > 1e-12:
                                        ratio = delta_meanabs / hidden_meanabs
                                    else:
                                        ratio = 0.0
                                except Exception:
                                    ctx_stats = "style_ctx: stats_error"
                                    hidden_stats = "hidden_after_style: stats_error"
                                    out_stats = "style_out: stats_error"
                                    delta_stats = "style_delta: stats_error"
                                    scale_val = 0.0
                                    ratio = 0.0

                                print(
                                    f"[AnimaAdapter] ✅ StyleAttn 已接入: "
                                    f"block={block_idx}, index={i}, "
                                    f"strength={strength}, "
                                    f"style_scale={scale_val:.8f}, "
                                    f"runtime_gain={runtime_gain:.4f}, "
                                    f"delta_to_hidden_meanabs_ratio={ratio:.8f}"
                                )
                                print("[AnimaAdapter]    " + ctx_stats)
                                print("[AnimaAdapter]    " + out_stats)
                                print("[AnimaAdapter]    " + delta_stats)
                                print("[AnimaAdapter]    " + hidden_stats)

                                if runtime_gain > 1.0:
                                    print(
                                        f"[AnimaAdapter] ℹ️ 当前 StyleAttn 使用了运行时补偿倍率: "
                                        f"{runtime_gain:.4f}。"
                                        f"如果画面风格过强、发脏或结构崩坏，请降低 Adapter strength "
                                        f"或降低 max_gain。"
                                    )

                                try:
                                    delta_absmax = float(style_delta.detach().float().abs().max().cpu())
                                    delta_meanabs = float(style_delta.detach().float().abs().mean().cpu())

                                    if delta_absmax < 1e-7 or delta_meanabs < 1e-8:
                                        print(
                                            "[AnimaAdapter] ⚠️ style_delta 仍然非常小。"
                                            "即使启用了 runtime_gain，Style 分支实际影响仍可能不足。"
                                            "建议继续训练 Adapter 或提高 Adapter strength。"
                                        )
                                except Exception:
                                    pass

        return hidden

    def _configure_style_runtime_boost(
            self,
            enable=True,
            target_to_out_absmax=2e-3,
            max_gain=128.0,
            min_problem_absmax=1e-4,
    ):
        """
        Runtime StyleAttn 输出补偿。

        作用：
        当 StyleAttn 的 to_out 权重过小时，style_out 会非常弱。
        这个函数不会修改 safetensors 文件，只是在运行时记录一个补偿倍率，
        后续 apply_block 里用 style_runtime_gain 放大 style_delta。

        参数：
        enable:
            是否启用自动补偿。
        target_to_out_absmax:
            期望补偿后的等效 to_out absmax。
            建议 1e-3 ~ 5e-3。
        max_gain:
            最大补偿倍率，防止炸图。
        min_problem_absmax:
            小于这个值时认为 to_out 偏弱。
        """

        self._style_runtime_gain = 1.0
        self._style_runtime_boost_enabled = bool(enable)

        if not enable:
            print("[AnimaAdapter] Style Runtime Boost: disabled")
            return

        style_count = int(getattr(self, "style_count", 0))
        if style_count <= 0 or not hasattr(self, "style_blocks"):
            print("[AnimaAdapter] Style Runtime Boost: no style blocks")
            return

        style_to_out_absmax = 0.0

        try:
            for blk in self.style_blocks:
                style_to_out_absmax = max(style_to_out_absmax, _module_param_absmax(blk.to_out))
        except Exception as e:
            print(f"[AnimaAdapter] Style Runtime Boost 检查失败: {e}")
            return

        self._style_to_out_absmax = float(style_to_out_absmax)

        if style_to_out_absmax <= 0:
            print(
                "[AnimaAdapter] ⚠️ Style Runtime Boost: to_out_absmax 为 0，"
                "无法通过倍率补偿修复。这个 Adapter 的 StyleAttn 输出层可能是坏权重。"
            )
            self._style_runtime_gain = 1.0
            return

        if style_to_out_absmax < min_problem_absmax:
            gain = float(target_to_out_absmax / style_to_out_absmax)
            gain = max(1.0, min(float(max_gain), gain))
            self._style_runtime_gain = gain

            print(
                f"[AnimaAdapter] ⚠️ 检测到 StyleAttn to_out 过小: "
                f"to_out_absmax={style_to_out_absmax:.8f}。"
            )
            print(
                f"[AnimaAdapter] ✅ 已启用 Style Runtime Boost: "
                f"target_to_out_absmax={target_to_out_absmax:.8f}, "
                f"runtime_gain={gain:.4f}, max_gain={max_gain:.1f}"
            )
        else:
            self._style_runtime_gain = 1.0
            print(
                f"[AnimaAdapter] Style Runtime Boost: 不需要补偿。"
                f"to_out_absmax={style_to_out_absmax:.8f}"
            )


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


def _normalize_style_embeds_for_runtime(style_embeds):
    if not torch.is_tensor(style_embeds):
        return None

    if style_embeds.ndim == 2:
        style_embeds = style_embeds.unsqueeze(1)
    elif style_embeds.ndim > 3:
        style_embeds = style_embeds.reshape(style_embeds.shape[0], -1, style_embeds.shape[-1])
    elif style_embeds.ndim != 3:
        raise RuntimeError(f"style_embeds 维度不支持: shape={tuple(style_embeds.shape)}")

    return style_embeds.detach()


def _style_tensor_signature(style_embeds):
    """
    用于判断 style_embeds 是否变化。

    之前只比较 shape/dtype，会导致：
    换了风格图，但 shape/dtype 一样 -> 系统误认为没变化 -> 继续使用旧 style_embeds。

    这里加入 data_ptr 和少量统计值，避免同尺寸风格图被误判为同一个。
    """
    if not torch.is_tensor(style_embeds):
        return None

    try:
        x = style_embeds.detach()
        shape = tuple(x.shape)
        dtype = str(x.dtype)
        device = str(x.device)
        ptr = int(x.data_ptr())

        # 少量统计值，避免 data_ptr 复用时误判。
        # style_embeds 一般不大，这个开销可以接受。
        xf = x.float()
        mean_val = float(xf.mean().detach().cpu())
        std_val = float(xf.std().detach().cpu()) if x.numel() > 1 else 0.0

        return shape, dtype, device, ptr, round(mean_val, 6), round(std_val, 6)
    except Exception:
        return tuple(style_embeds.shape), str(style_embeds.dtype), str(style_embeds.device)


def _adapter_stack_signature(adapter_stack, device="auto", dtype="auto", num_blocks=28, style_embeds=None):
    signature = []

    for item in adapter_stack or []:
        name = item.get("name", "None")
        strength = float(item.get("strength", 1.0))

        if not name or name == "None" or abs(strength) < 1e-6:
            continue

        signature.append((name, strength))

    style_sig = _style_tensor_signature(style_embeds)

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

    if hasattr(diffusion_model, "blocks"):
        actual_num_blocks = len(diffusion_model.blocks)
        if int(num_blocks) != int(actual_num_blocks):
            print(
                f"[AnimaAdapter] ⚠️ adapter_num_blocks={num_blocks} 与底模真实 blocks={actual_num_blocks} 不一致。"
                f"将使用真实 blocks={actual_num_blocks}，避免 local/style 起始层错位。"
            )
            num_blocks = int(actual_num_blocks)
    else:
        actual_num_blocks = int(num_blocks)

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

    style_embeds = _normalize_style_embeds_for_runtime(style_embeds)

    if torch.is_tensor(style_embeds):
        print(
            f"[AnimaAdapter] 已接收 style_embeds: "
            f"shape={tuple(style_embeds.shape)}, dtype={style_embeds.dtype}, device={style_embeds.device}"
        )
        print("[AnimaAdapter] " + _safe_tensor_stats_str(style_embeds, "received_style_embeds"))

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
            print("[AnimaAdapter] Adapter 配置未变化，沿用已挂载插件")
            return model_patcher

        if old_signature is not None and len(old_signature) == 5:
            old_core = old_signature[:4]
            new_core = signature[:4]

            if old_core == new_core:
                model_patcher._anima_runtime_style_embeds = style_embeds
                model_patcher._anima_runtime_adapter_signature = signature
                print("[AnimaAdapter] 检测到仅 style_embeds 变化，已更新运行时 style_embeds")
                return model_patcher

        raise RuntimeError(
            "当前 MODEL 已经挂载过另一组 Anima Adapter。"
            "请重新执行模型烧录器节点，或重新加载工作流后再更换 Adapter / 强度 / dtype。"
        )

    runtime_adapters = []

    for item in active_stack:
        name = item.get("name", "None")
        strength = float(item.get("strength", 1.0))

        path = folder_paths.get_full_path(ANIMA_ADAPTER_FOLDER_NAME, name)
        if path is None:
            print(f"[AnimaAdapter] 找不到插件: {name}")
            continue

        print(f"[AnimaAdapter] 加载插件: {path} | strength={strength}")

        sd = comfy.utils.load_torch_file(path, safe_load=True)

        file_dtype = _floating_dtype_from_state_dict(sd, None)
        runtime_dtype = _resolve_adapter_runtime_dtype(dtype, file_dtype, model_dtype)

        print(
            f"   -> 文件 dtype: {_dtype_name(file_dtype)} | "
            f"底模 dtype: {_dtype_name(model_dtype)} | "
            f"运行 dtype: {_dtype_name(runtime_dtype)}"
        )

        adapter = RuntimeAnimaAdapter.from_state_dict(sd, num_blocks=num_blocks)
        adapter.to(device=target_device, dtype=runtime_dtype)
        adapter.eval()

        param_count = sum(p.numel() for p in adapter.parameters())
        bytes_per_param = torch.tensor([], dtype=runtime_dtype).element_size()
        size_mb = param_count * bytes_per_param / 1024 / 1024

        print(
            f"   -> 参数量: {param_count:,} | "
            f"约 {size_mb:.2f} MB ({runtime_dtype})"
        )

        runtime_adapters.append((adapter, strength))

        del sd
        gc.collect()

    if not runtime_adapters:
        print("[AnimaAdapter] 没有有效插件")
        return model_patcher

    if not hasattr(diffusion_model, "blocks"):
        raise RuntimeError("Anima DiT 不存在 blocks，无法挂载插件")

    print(f"[AnimaAdapter] 底模 blocks 数量: {len(diffusion_model.blocks)}")

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
                if not getattr(model_patcher, "_anima_first_block_called_printed", False):
                    print(
                        f"[AnimaAdapter] ✅ block.forward wrapper 已生效: "
                        f"first_block={_idx}, hidden_shape={tuple(hidden.shape)}, "
                        f"hidden_dtype={hidden.dtype}, hidden_device={hidden.device}"
                    )
                    model_patcher._anima_first_block_called_printed = True

                current_style_embeds = getattr(
                    model_patcher,
                    "_anima_runtime_style_embeds",
                    None,
                )

                for adapter, strength in runtime_adapters:
                    hidden = adapter.apply_block(
                        _idx,
                        hidden,
                        text_context=text_context,
                        style_context=current_style_embeds,
                        strength=strength,
                    )
            else:
                if not getattr(model_patcher, "_anima_hidden_not_tensor_warned", False):
                    print(
                        f"[AnimaAdapter] ⚠️ block.forward 输出不是 tensor，Adapter 无法作用。"
                        f"block={_idx}, type={type(hidden)}"
                    )
                    model_patcher._anima_hidden_not_tensor_warned = True

            if is_tuple:
                return (hidden,) + tuple(out[1:])

            return hidden

        block.forward = types.MethodType(new_forward, block)

    model_patcher._anima_runtime_adapter_patched = True
    model_patcher._anima_runtime_adapter_signature = signature
    model_patcher._anima_runtime_adapters = runtime_adapters
    model_patcher._anima_runtime_style_embeds = style_embeds

    print(f"[AnimaAdapter] 已挂载 {len(runtime_adapters)} 个 Adapter 插件")
    return model_patcher


class AnimaStyleEmbedsFromClipVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "target_dim": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 1}),
                "normalize": ("BOOLEAN", {"default": False}),
                "source": (
                    [
                        "auto_nonzero",
                        "image_embeds",
                        "pooled_output",
                        "penultimate_hidden_states",
                        "last_hidden_state",
                        "cond",
                    ],
                    {"default": "auto_nonzero"},
                ),
                "token_handling": (
                    ["keep_tokens", "mean_pool", "cls_token"],
                    {"default": "keep_tokens"},
                ),
                "dim_mismatch_policy": (
                    ["warn_crop_or_pad", "error", "keep_original"],
                    {"default": "warn_crop_or_pad"},
                ),
                "print_debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ANIMA_STYLE_EMBEDS",)
    RETURN_NAMES = ("style_embeds",)
    FUNCTION = "make"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def make(
            self,
            clip_vision_output,
            target_dim=768,
            normalize=False,
            source="auto_nonzero",
            token_handling="keep_tokens",
            dim_mismatch_policy="warn_crop_or_pad",
            print_debug=True,
    ):
        preferred_key = None if source == "auto_nonzero" else source

        x = _extract_clip_vision_tensor(
            clip_vision_output,
            preferred_key=preferred_key,
            allow_zero=False,
            verbose=bool(print_debug),
        )

        if x is None:
            raise RuntimeError(
                "[AnimaStyle] 无法从 CLIP_VISION_OUTPUT 中提取有效的非零 tensor。"
                "这通常说明 CLIP Vision Encode 节点输出异常，或者当前代码取到的字段全为 0。"
                "请连接 AnimaClipVisionInspector 节点检查各字段。"
            )

        if not torch.is_tensor(x):
            raise RuntimeError("[AnimaStyle] CLIP_VISION_OUTPUT 提取结果不是 tensor。")

        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        original_device = x.device

        x = x.detach().float()

        if x.ndim == 2:
            x = x.unsqueeze(1)

        elif x.ndim == 3:
            if token_handling == "mean_pool":
                x = x.mean(dim=1, keepdim=True)
            elif token_handling == "cls_token":
                x = x[:, :1, :]
            elif token_handling == "keep_tokens":
                pass
            else:
                raise RuntimeError(f"[AnimaStyle] 不支持的 token_handling: {token_handling}")

        elif x.ndim == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])

            if token_handling == "mean_pool":
                x = x.mean(dim=1, keepdim=True)
            elif token_handling == "cls_token":
                x = x[:, :1, :]
            elif token_handling == "keep_tokens":
                pass
            else:
                raise RuntimeError(f"[AnimaStyle] 不支持的 token_handling: {token_handling}")

        else:
            raise RuntimeError(
                f"[AnimaStyle] style_embeds 维度不支持: "
                f"original_shape={original_shape}, current_shape={tuple(x.shape)}"
            )

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"[AnimaStyle] 提取到的 style tensor 存在 NaN 或 Inf: "
                f"original_shape={original_shape}, current_shape={tuple(x.shape)}"
            )

        source_dim = int(x.shape[-1])
        target_dim = int(target_dim)

        pre_absmax = float(x.abs().max().detach().cpu()) if x.numel() > 0 else 0.0
        pre_std = float(x.std().detach().cpu()) if x.numel() > 1 else 0.0

        if pre_absmax == 0.0 or pre_std == 0.0:
            raise RuntimeError(
                f"[AnimaStyle] 提取到的 style_embeds 是全 0 或无方差，无法提供风格信息。\n"
                f"source={source}, original_shape={original_shape}, "
                f"current_shape={tuple(x.shape)}, dtype={original_dtype}, device={original_device}\n"
                f"请检查：\n"
                f"1. CLIP Vision Encode 节点是否真的接收到了风格图。\n"
                f"2. 是否用了错误的 CLIP Vision 输出字段。\n"
                f"3. 是否有其它节点把 CLIP Vision 输出清零。\n"
                f"4. 请使用 AnimaClipVisionInspector 节点查看各字段统计。"
            )

        if normalize:
            x = F.normalize(x, dim=-1)

        if dim_mismatch_policy == "keep_original":
            final_target_dim = source_dim
        else:
            if target_dim <= 0:
                final_target_dim = source_dim
            else:
                final_target_dim = target_dim

            if source_dim != final_target_dim:
                msg = (
                    f"[AnimaStyle] 警告：CLIP Vision 输出维度与目标维度不一致: "
                    f"source_dim={source_dim}, target_dim={final_target_dim}。\n"
                    f"当前会执行 {'裁剪' if source_dim > final_target_dim else '补零'}，"
                    f"但这不是学习过的投影，风格迁移效果可能变弱。\n"
                    f"如果 Adapter 结构显示 style_dim=768，请使用 OpenAI CLIP ViT-L/14。\n"
                    f"如果 Adapter 结构显示 style_dim=1024，请使用对应 1024 维视觉编码器，"
                    f"并把 target_dim 设置为 1024。"
                )

                if dim_mismatch_policy == "error":
                    raise RuntimeError(msg)

                print(msg)

            x = _match_last_dim(x, final_target_dim)

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"[AnimaStyle] style_embeds 出现 NaN 或 Inf: "
                f"original_shape={original_shape}, final_shape={tuple(x.shape)}"
            )

        mean_val = float(x.float().mean().detach().cpu()) if x.numel() > 0 else 0.0
        std_val = float(x.float().std().detach().cpu()) if x.numel() > 1 else 0.0
        min_val = float(x.float().min().detach().cpu()) if x.numel() > 0 else 0.0
        max_val = float(x.float().max().detach().cpu()) if x.numel() > 0 else 0.0
        absmax_val = float(x.float().abs().max().detach().cpu()) if x.numel() > 0 else 0.0

        if absmax_val == 0.0 or std_val == 0.0:
            raise RuntimeError(
                f"[AnimaStyle] 最终 style_embeds 仍然是全 0 或无方差，已终止。\n"
                f"final_shape={tuple(x.shape)}, mean={mean_val}, std={std_val}, absmax={absmax_val}"
            )

        print(
            f"🎨 [AnimaStyle] style_embeds 已生成 | "
            f"source={source} | token_handling={token_handling} | "
            f"original_shape={original_shape} | final_shape={tuple(x.shape)} | "
            f"source_dim={source_dim} | target_dim={x.shape[-1]} | "
            f"dtype={x.dtype} | "
            f"mean={mean_val:.6f} | std={std_val:.6f} | "
            f"min={min_val:.6f} | max={max_val:.6f} | absmax={absmax_val:.6f}"
        )

        return (x,)


class AnimaClipVisionInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "print_to_console": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def inspect(self, clip_vision_output, print_to_console=True):
        candidates = _collect_clip_vision_tensors(clip_vision_output)

        lines = []
        lines.append("[AnimaClipVisionInspector] CLIP Vision 输出检查")

        if not candidates:
            lines.append("没有找到任何 tensor。")
            report = "\n".join(lines)
            if print_to_console:
                print(report)
            return (report,)

        for name, tensor in candidates:
            st = _tensor_basic_stats(tensor)

            lines.append(
                f"- {name}: "
                f"shape={st['shape']}, "
                f"dtype={st['dtype']}, "
                f"device={st['device']}, "
                f"finite={st['finite']}, "
                f"all_zero={st['all_zero']}, "
                f"mean={st['mean']:.8f}, "
                f"std={st['std']:.8f}, "
                f"min={st['min']:.8f}, "
                f"max={st['max']:.8f}, "
                f"absmax={st['absmax']:.8f}"
            )

        report = "\n".join(lines)

        if print_to_console:
            print(report)

        return (report,)


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
    "AnimaClipVisionInspector": AnimaClipVisionInspector,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterStack": "Anima Adapter 插件堆",
    "AnimaDeviceFix": "Anima 设备修复器",
    "AnimaStyleEmbedsFromClipVision": "Anima Style Embeds From CLIP Vision",
    "AnimaClipVisionInspector": "Anima CLIP Vision Inspector",
    "SeparateModelMixerDictFuser": "Anima模型烧录器",
}
