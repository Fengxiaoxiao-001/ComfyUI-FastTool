import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.lora
import torch
import gc
from collections import OrderedDict
from hashlib import sha256


class TrueModelMixerDictFuser:
    _cache = OrderedDict()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),),
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
    CATEGORY = "Advanced LoRA/Fusion"

    def pure_dict_merge(self, base_model, lora_stack=None, save_dtype="auto"):
        if lora_stack is None:
            lora_stack = []

        active_loras = [l for l in lora_stack if l[0] and l[0] != "None" and (abs(l[1]) > 0.0001 or abs(l[2]) > 0.0001)]
        lora_key = "|".join([f"{n}-{sm}-{sc}" for n, sm, sc in active_loras]) if active_loras else "none"
        cache_id = sha256(f"{base_model}|{lora_key}|{save_dtype}".encode()).hexdigest()

        if cache_id in self._cache:
            print("📦 [TrueMixer] 从缓存中读取融合模型...")
            return self._cache[cache_id]

        print("🧹 [TrueMixer] 准备开始烧录，正在强制清空残留显存...")

        comfy.model_management.unload_all_models()

        gc.collect()

        comfy.model_management.soft_empty_cache()

        print(f"📦 [TrueMixer] 正在加载底模: {base_model}")
        ckpt_path = folder_paths.get_full_path("checkpoints", base_model)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip, vae = out[0], out[1], out[2]

        base_dtype = next(model.model.parameters()).dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["fp8_e4m3fn"] = torch.float8_e4m3fn
            dtype_map["fp8_e5m2"] = torch.float8_e5m2

        target_dtype = dtype_map.get(save_dtype, base_dtype)

        if not active_loras and target_dtype == base_dtype:
            return (model, clip, vae)

        print(f"🛠️ [TrueMixer] 正在解析 {len(active_loras)} 个 LoRA 补丁...")
        for lora_name, m_strength, c_strength in active_loras:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path:
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, m_strength, c_strength)

        @torch.inference_mode()
        def bake_model_weights(patcher, name="MODEL"):
            print(f"🔥 [TrueMixer] 正在以 {target_dtype} 精度烧录 {name}...")
            device = comfy.model_management.get_torch_device()
            model_inner = patcher.model
            model_inner.to(device)

            sd = model_inner.state_dict()
            new_sd = {}

            for key in sd:

                weight = sd[key].to(device).to(torch.float32)

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

            model_inner.to(comfy.model_management.intermediate_device())
            return patcher

        model = bake_model_weights(model, "UNET")
        if hasattr(clip, "patcher"):
            clip.patcher = bake_model_weights(clip.patcher, "CLIP")

        gc.collect()
        comfy.model_management.soft_empty_cache()

        results = (model, clip, vae)
        self._cache[cache_id] = results
        if len(self._cache) > 2: self._cache.popitem(last=False)

        return results


NODE_CLASS_MAPPINGS = {"TrueModelMixerDictFuser": TrueModelMixerDictFuser}
NODE_DISPLAY_NAME_MAPPINGS = {"TrueModelMixerDictFuser": "【SDXL】 Model Mixer"}
