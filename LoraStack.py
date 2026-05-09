import folder_paths


class MultiLoRAStack:
    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        lora_choices = ["None"] + lora_list

        arg_dict = {
            "required": {},
            "optional": {
                "lora_stack": ("LORA_STACK",),
            }
        }

        for i in range(1, 5):
            arg_dict["optional"][f"lora_{i}"] = (lora_choices, {"default": "None"})

            arg_dict["optional"][f"model_strength_{i}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "UNET 权重"
            })
            arg_dict["optional"][f"clip_strength_{i}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "CLIP 权重"
            })

        return arg_dict

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "custom/LoRA Utils"
    DESCRIPTION = "多 LoRA 堆叠器：支持 4 个插槽，只有选择了具体 LoRA 文件且权重不全为 0 时才会生效。"

    def stack(self, lora_stack=None, **kwargs):
        result = []

        if lora_stack is not None and isinstance(lora_stack, list):
            result.extend(lora_stack)

        for i in range(1, 5):
            lora_name = kwargs.get(f"lora_{i}")

            if lora_name and lora_name != "None":
                sm = kwargs.get(f"model_strength_{i}", 1.0)
                sc = kwargs.get(f"clip_strength_{i}", 1.0)

                if abs(sm) > 0.0001 or abs(sc) > 0.0001:
                    result.append((lora_name, sm, sc))
        return (result,)


# ====================== 注册字典 ======================
NODE_CLASS_MAPPINGS = {
    "MultiLoRAStack": MultiLoRAStack
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器"
}
