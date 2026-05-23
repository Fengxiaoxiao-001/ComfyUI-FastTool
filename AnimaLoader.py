import folder_paths
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import comfy.model_management
import comfy.lora
from comfy.sd import CLIP
import torch
import gc
from comfy.text_encoders.anima import AnimaTEModel


class SeparateModelMixerDictFuser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"),),
                "clip": (folder_paths.get_filename_list("clip"),),
                "vae": (folder_paths.get_filename_list("vae"),),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "save_dtype": (["auto", "float16", "bfloat16", "float32"],
                               {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "pure_dict_merge"
    CATEGORY = "XiaoXiao/Fusion[Anima]"

    def pure_dict_merge(self, model, clip, vae, lora_stack=None, save_dtype="auto", device="auto"):
        if lora_stack is None:
            lora_stack = []

        active_loras = [l for l in lora_stack if l[0] and l[0] != "None" and (abs(l[1]) > 0.0001 or abs(l[2]) > 0.0001)]

        print("🧹 [TrueMixer] 清空残留显存...")
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()

        if device == "auto":
            target_device = comfy.model_management.get_torch_device()
        else:
            target_device = torch.device(device)

        print(f"🔧 [TrueMixer] 使用设备: {target_device}")

        def find_model_path(model_name, folder_types):
            """在多个文件夹类型中搜索模型"""
            for folder_type in folder_types:
                path = folder_paths.get_full_path(folder_type, model_name)
                if path is not None:
                    return path, folder_type
            return None, None

        model_path, model_folder = find_model_path(model, ["checkpoints"])
        if model_path is None:
            raise RuntimeError(
                f"找不到主模型文件: {model}\n"
                f"请确保文件在以下目录之一:\n"
                f"  - models/checkpoints/\n"
                f"  - models/diffusion_models/\n"
                f"  - models/unet/"
            )
        print(f"📥 [TrueMixer] 找到主模型: {model_path} (来自 {model_folder})")

        vae_path, vae_folder = find_model_path(vae, ["vae"])
        if vae_path is None:
            raise RuntimeError(f"找不到VAE模型文件: {vae}\n请确保文件在 models/vae/ 目录下")
        print(f"📥 [TrueMixer] 找到VAE: {vae_path}")

        clip_path, clip_folder = find_model_path(clip, ["clip"])
        if clip_path is None:
            raise RuntimeError(
                f"找不到CLIP模型文件: {clip}\n"
                f"请确保文件在以下目录之一:\n"
                f"  - models/clip/\n"
                f"  - models/text_encoders/"
            )
        print(f"📥 [TrueMixer] 找到CLIP: {clip_path} (来自 {clip_folder})")

        print(f"📥 [TrueMixer] 加载 Anima 基础模型: {model_path}")
        ckpt_out = comfy.sd.load_checkpoint_guess_config(
            model_path,
            output_vae=False,
            output_clip=False
        )

        model_obj = ckpt_out[0]

        print(f"📥 [TrueMixer] 加载 Anima VAE: {vae_path}")
        sd = comfy.utils.load_torch_file(vae_path)
        vae_obj = comfy.sd.VAE(sd=sd)

        base_dtype = next(model_obj.model.parameters()).dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["fp8_e4m3fn"] = torch.float8_e4m3fn
            dtype_map["fp8_e5m2"] = torch.float8_e5m2

        target_dtype = dtype_map.get(save_dtype, base_dtype)

        print(f"📥 [TrueMixer] 加载 Anima Qwen3 CLIP: {clip_path}")
        clip_sd = comfy.utils.load_torch_file(clip_path, safe_load=True)

        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=comfy.text_encoders.anima.AnimaTokenizer,
            clip=AnimaTEModel
        )

        clip_obj = CLIP(clip_target, embedding_directory=None)
        clip_obj.load_sd(clip_sd)

        del clip_sd
        gc.collect()

        if not active_loras and target_dtype == base_dtype:
            return (model_obj, clip_obj, vae_obj)

        model_clone = model_obj.clone()
        if hasattr(clip_obj, 'clone'):
            clip_clone = clip_obj.clone()
        else:
            import copy
            clip_clone = copy.copy(clip_obj)

        print(f"🛠️ [TrueMixer] 正在解析 {len(active_loras)} 个 LoRA 补丁...")
        for lora_name, m_strength, c_strength in active_loras:
            lora_path, _ = find_model_path(lora_name, ["loras"])
            if lora_path:
                print(f"  - 应用 LoRA: {lora_name} (模型强度: {m_strength}, CLIP强度: {c_strength})")
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

                if hasattr(clip_clone, 'cond_stage_model'):
                    model_clone, clip_clone = comfy.sd.load_lora_for_models(
                        model_clone, clip_clone, lora_sd, m_strength, c_strength
                    )
                else:

                    model_clone, _ = comfy.sd.load_lora_for_models(
                        model_clone, None, lora_sd, m_strength, c_strength
                    )
                del lora_sd
                gc.collect()

        @torch.inference_mode()
        def bake_model_weights(patcher, name="MODEL", force_cpu=False):

            calc_device = torch.device('cpu') if force_cpu else target_device

            print(f"🔥 [TrueMixer] 正在使用设备 [{calc_device}] 以 {target_dtype} 精度烧录 {name}...")

            model_inner = patcher.model
            model_inner.to(calc_device)

            sd = model_inner.state_dict()
            new_sd = {}

            total_layers = len(sd)
            for i, key in enumerate(sd):
                if i % 100 == 0:
                    print(f"  - 处理层 {i + 1}/{total_layers}...")

                weight = sd[key].to(calc_device).to(torch.float32)

                if key in patcher.patches:
                    weight = comfy.lora.calculate_weight(patcher.patches[key], weight, key)

                if target_dtype == torch.float16:
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=65504, neginf=-65504)
                elif "float8" in str(target_dtype):
                    max_val = 448.0 if target_dtype == torch.float8_e4m3fn else 57344.0
                    weight = torch.clamp(weight, -max_val, max_val)
                    weight = torch.nan_to_num(weight, nan=0.0)

                new_sd[key] = weight.to(target_dtype).cpu()

            patcher.patches = {}
            patcher.backup = {}
            if hasattr(patcher, 'object_patches'):
                patcher.object_patches = {}

            model_inner.load_state_dict(new_sd)

            if hasattr(patcher, "base_model"):
                patcher.base_model.load_state_dict(new_sd)

            if force_cpu:
                model_inner.to(torch.device('cpu'))
                print(f"🧠 [TrueMixer] {name} 烧录完成。全程未进入显存，目前安全驻留在系统内存(RAM)中。")
            else:
                model_inner.to(comfy.model_management.intermediate_device())

            return patcher

        model_clone = bake_model_weights(model_clone, "UNET/Transformer", force_cpu=(device == "cpu"))

        if hasattr(clip_clone, "patcher") and clip_clone.patcher is not None:
            print("🔥 [TrueMixer] 正在烧录 CLIP...")
            clip_clone.patcher = bake_model_weights(clip_clone.patcher, "CLIP", force_cpu=(device == "cpu"))
        else:
            print("⚠️ [TrueMixer] CLIP 没有 patcher，跳过烧录")

        gc.collect()
        comfy.model_management.soft_empty_cache()
        print("✅ [TrueMixer] 纯净烧录全部结束！")

        return (model_clone, clip_clone, vae_obj)


NODE_CLASS_MAPPINGS = {"SeparateModelMixerDictFuser": SeparateModelMixerDictFuser}
NODE_DISPLAY_NAME_MAPPINGS = {"SeparateModelMixerDictFuser": "️Anima模型烧录器"}
