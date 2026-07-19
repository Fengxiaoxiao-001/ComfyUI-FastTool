import folder_paths


class MultiLoRAStack:
    @classmethod
    def INPUT_TYPES(s):
        # 获取现有的 lora 文件列表
        lora_list = folder_paths.get_filename_list("loras")
        # 核心修复：在列表首位插入 "None"，确保下拉菜单有“不填”的选项
        lora_choices = ["None"] + lora_list

        arg_dict = {
            "required": {},
            "optional": {
                # 允许链接另一个堆叠节点，实现 4+4+4... 串联
                "lora_stack": ("LORA_STACK",),
            }
        }

        # 动态创建 4 个插槽
        for i in range(1, 5):
            # 1. 使用含 "None" 的列表
            # 2. 设置 default 为 "None"，这样 UI 初始化时会自动选中它，不会因为“没填”而报错
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
    CATEGORY = "XiaoXiao/LoRA Utils"
    DESCRIPTION = "多 LoRA 堆叠器：支持 4 个插槽，只有选择了具体 LoRA 文件且权重不全为 0 时才会生效。"

    def stack(self, lora_stack=None, **kwargs):
        # 存放最终有效 LoRA 的列表
        result = []

        # 1. 优先继承上游堆叠的数据
        if lora_stack is not None and isinstance(lora_stack, list):
            result.extend(lora_stack)

        # 2. 遍历当前节点的 4 个插槽
        for i in range(1, 5):
            lora_name = kwargs.get(f"lora_{i}")

            # 核心逻辑：只有名字不是 "None" 时才处理
            # 如果你只填了第 1 个，第 2,3,4 个因为默认是 "None" 会被直接跳过
            if lora_name and lora_name != "None":
                sm = kwargs.get(f"model_strength_{i}", 1.0)
                sc = kwargs.get(f"clip_strength_{i}", 1.0)

                # 只有当 UNET 和 CLIP 权重【全为 0】时才过滤
                # 如果你设置 Clip 为 0 但 UNET 为 1，它依然会正常进入 result
                if abs(sm) > 0.0001 or abs(sc) > 0.0001:
                    result.append((lora_name, sm, sc))

        # 即使 result 是空的 []，也会返回给下游，不会报错
        return (result,)


# ====================== 注册字典 ======================
NODE_CLASS_MAPPINGS = {
    "MultiLoRAStack": MultiLoRAStack
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器"
}
