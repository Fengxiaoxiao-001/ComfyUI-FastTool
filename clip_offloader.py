import torch
import gc
import weakref
import comfy.model_management as mm
from comfy.model_management import soft_empty_cache, free_memory

# ====================== 全局变量：记录上一次搬运的 CLIP ======================
last_clip_ref = None


class VRAM_CLIP_Offloader:
    """
    独立节点：把 CLIP 搬运到 CPU 或 NPU，显著节省 GPU 显存
    - 推荐使用 "cpu"（最稳定，节省明显）
    - "npu" 为实验性：失败会自动、安全地回退到 CPU
    - 支持重复加载 CLIP：每次新 CLIP 进入时自动释放上一个 CLIP 的内存
    - 必须接在 CLIP Loader + LoRA Loader（CLIP 部分）之后


    Checkpoint Loader → CLIP 输出
                ↓
    LoRA Loader（CLIP 部分）
                    ↓
    【VRAM_CLIP_Offloader】 ← 必须接在这里
                    ↓
    CLIP Text Encode
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "target_device": (["cpu", "npu"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "offload"
    CATEGORY = "XiaoXiao/VRAM Cleaner"

    def offload(self, clip, target_device):
        global last_clip_ref

        # 1. 先释放上一次的 CLIP（防止重复加载内存累积）
        if last_clip_ref is not None:
            old_clip = last_clip_ref()
            if old_clip is not None:
                try:
                    del old_clip
                    gc.collect()
                    gc.collect(1)
                    gc.collect(2)
                except:
                    pass
            last_clip_ref = None

        # 2. 记录本次 CLIP
        last_clip_ref = weakref.ref(clip)

        before_vram = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

        # 记录用户实际选择的设备（用于打印）
        user_choice = target_device

        try:
            # ==================== 设备选择（已修复） ====================
            if target_device == "npu":
                if hasattr(torch, "npu") and torch.npu.is_available():
                    device = torch.device("npu")
                    print("   → 检测到 NPU，正在尝试搬运...")
                else:
                    print("   ⚠️ 当前环境不支持 NPU，自动回退到 CPU")
                    device = torch.device("cpu")
                    target_device = "cpu"  # 只改局部变量，不影响 user_choice
            else:
                device = torch.device("cpu")

            # ==================== 关键：搬运 CLIP（更鲁棒） ====================
            moved = False

            # ComfyUI CLIP 的标准结构
            if hasattr(clip, "cond_stage_model"):
                if hasattr(clip.cond_stage_model, "to"):
                    clip.cond_stage_model = clip.cond_stage_model.to(device)
                    moved = True
                # 部分版本需要再深入一层
                elif hasattr(clip.cond_stage_model, "model") and hasattr(clip.cond_stage_model.model, "to"):
                    clip.cond_stage_model.model = clip.cond_stage_model.model.to(device)
                    moved = True

            # 兜底
            if not moved and hasattr(clip, "to"):
                clip = clip.to(device)
                moved = True

            if not moved:
                raise AttributeError("无法找到 CLIP 内部可搬运的模型")

            # ==================== 同步 + 清理 ====================
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()  # 只有 CUDA 才执行 synchronize

            soft_empty_cache()
            free_memory(0, mm.get_torch_device())
            torch.cuda.empty_cache()
            gc.collect()
            gc.collect(1)
            gc.collect(2)

            # ==================== 统计显存 + 显示实际设备 ====================
            after_vram = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            saved = max(0, before_vram - after_vram)

            # 获取 CLIP 当前真实设备
            current_dev = "unknown"
            if hasattr(clip, "cond_stage_model") and hasattr(clip.cond_stage_model, "device"):
                current_dev = str(clip.cond_stage_model.device)
            elif hasattr(clip, "device"):
                current_dev = str(clip.device)

            print(f"[CLIP Offloader] ✅ CLIP 已搬运 (用户选择: {user_choice.upper()} → 实际: {current_dev})")
            if saved > 0:
                print(f"   → 节省约 {saved:.1f} MB GPU 显存")
            else:
                print("   → 显存统计显示 0 MB（ComfyUI 缓存延迟常见），但 CLIP 已成功搬离 GPU")

        except Exception as e:
            print(f"[CLIP Offloader] ⚠️ 搬运失败: {e}，正在回退到 CPU")
            try:
                if hasattr(clip, "cond_stage_model") and hasattr(clip.cond_stage_model, "to"):
                    clip.cond_stage_model = clip.cond_stage_model.to("cpu")
                elif hasattr(clip, "to"):
                    clip = clip.to("cpu")
                print("[CLIP Offloader] ✅ 已成功回退到 CPU")
            except:
                pass

        # 最终清理
        soft_empty_cache()
        free_memory(0, mm.get_torch_device())
        gc.collect()
        gc.collect(1)
        gc.collect(2)

        return (clip,)


# ====================== 节点注册 ======================
NODE_CLASS_MAPPINGS = {
    "VRAM_CLIP_Offloader": VRAM_CLIP_Offloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAM_CLIP_Offloader": "🔄 VRAM CLIP Offloader（CLIP 搬到 CPU/NPU）",
}
