# Anima Adapter 插件使用指南 | Anima Adapter Plugin User Guide

## 简介 | Introduction
这个ComfyUI插件为Anima DiT系列模型提供了轻量级的Adapter支持，允许你在不修改基础模型权重的情况下，快速添加风格、主题、细节增强等效果。插件还包含一个强大的模型烧录器，可以将LoRA和Adapter效果永久烧录到模型中，方便分享和使用。

This ComfyUI plugin provides lightweight Adapter support for the Anima DiT series of models, allowing you to quickly add styles, themes, detail enhancements, and other effects without modifying the base model weights. The plugin also includes a powerful model burner that can permanently bake LoRA and Adapter effects into the model for easy sharing and use.

## 安装指南 | Installation Guide
1. 将本插件文件夹放入ComfyUI的`custom_nodes`目录中
2. 重启ComfyUI服务
3. 将下载的Anima Adapter文件放入`models/anima_adapters`目录中
4. 刷新ComfyUI网页界面

1. Place this plugin folder into the `custom_nodes` directory of ComfyUI
2. Restart the ComfyUI service
3. Place downloaded Anima Adapter files into the `models/anima_adapters` directory
4. Refresh the ComfyUI web interface

## 节点详细说明 | Node Detailed Description

### 1. Anima Adapter 插件堆 | Anima Adapter Stack
**功能描述 | Function Description**  
用于堆叠多个Anima Adapter插件，支持最多4个Adapter同时生效，也可以与之前的Adapter堆进行叠加。

Used to stack multiple Anima Adapter plugins, supporting up to 4 Adapters to take effect simultaneously, and can also be stacked with previous Adapter stacks.

**输入参数 | Input Parameters**
| 参数名 | Parameter Name | 类型 | Type | 默认值 | Default Value | 取值范围 | Value Range | 说明 | Description |
|--------|----------------|------|------|--------|---------------|----------|-------------|------|-------------|
| adapter_1 | adapter_1 | 下拉菜单 | Dropdown | None | 所有可用的Adapter文件 | All available Adapter files | 第一个要应用的Adapter | The first Adapter to apply |
| strength_1 | strength_1 | 浮点数 | Float | 1.0 | -5.0 ~ 5.0 | 第一个Adapter的强度，负值表示反向效果 | Strength of the first Adapter, negative values indicate reverse effect |
| adapter_2 | adapter_2 | 下拉菜单 | Dropdown | None | 所有可用的Adapter文件 | All available Adapter files | 第二个要应用的Adapter | The second Adapter to apply |
| strength_2 | strength_2 | 浮点数 | Float | 1.0 | -5.0 ~ 5.0 | 第二个Adapter的强度 | Strength of the second Adapter |
| adapter_3 | adapter_3 | 下拉菜单 | Dropdown | None | 所有可用的Adapter文件 | All available Adapter files | 第三个要应用的Adapter | The third Adapter to apply |
| strength_3 | strength_3 | 浮点数 | Float | 1.0 | -5.0 ~ 5.0 | 第三个Adapter的强度 | Strength of the third Adapter |
| adapter_4 | adapter_4 | 下拉菜单 | Dropdown | None | 所有可用的Adapter文件 | All available Adapter files | 第四个要应用的Adapter | The fourth Adapter to apply |
| strength_4 | strength_4 | 浮点数 | Float | 1.0 | -5.0 ~ 5.0 | 第四个Adapter的强度 | Strength of the fourth Adapter |
| prev_stack | prev_stack | ANIMA_ADAPTER_STACK | ANIMA_ADAPTER_STACK | None | 之前的Adapter堆 | Previous Adapter stack | 可选，与当前选择的Adapter叠加 | Optional, stacked with currently selected Adapters |

**输出 | Output**
- `anima_adapter_stack`: 组合后的Adapter堆，可以传递给其他节点使用
- `anima_adapter_stack`: Combined Adapter stack that can be passed to other nodes for use

**使用注意事项 | Usage Notes**
- Adapter的应用顺序是从下到上（prev_stack先应用，然后是adapter_1到adapter_4）
- 强度为0或选择"None"的Adapter会被自动忽略
- 多个Adapter叠加时，效果会相互影响，建议从低强度开始尝试

- Adapters are applied from bottom to top (prev_stack first, then adapter_1 to adapter_4)
- Adapters with strength 0 or selected as "None" are automatically ignored
- When multiple Adapters are stacked, effects will interact with each other. It is recommended to start with low strength

---

### 2. Anima 设备修复器 | Anima Device Fix
**功能描述 | Function Description**  
修复Anima模型中llm_adapter设备不匹配的问题，确保模型在正确的设备上运行，避免出现"tensor on different devices"错误。

Fixes the device mismatch issue of the llm_adapter in Anima models, ensuring the model runs on the correct device and avoiding "tensor on different devices" errors.

**输入参数 | Input Parameters**
| 参数名 | Parameter Name | 类型 | Type | 默认值 | Default Value | 取值范围 | Value Range | 说明 | Description |
|--------|----------------|------|------|--------|---------------|----------|-------------|------|-------------|
| model | model | MODEL | MODEL | - | 加载的Anima模型 | Loaded Anima model | 需要修复设备的Anima模型 | Anima model that needs device fixing |
| device | device | 下拉菜单 | Dropdown | auto | auto, cpu, cuda, npu | 目标运行设备 | Target running device |
| dtype | dtype | 下拉菜单 | Dropdown | auto | auto, from_file, float16, bfloat16, float32 | 目标数据类型 | Target data type |

**输出 | Output**
- `MODEL`: 修复后的Anima模型
- `MODEL`: Fixed Anima model

**使用注意事项 | Usage Notes**
- 建议在加载Anima模型后立即使用此节点
- "auto"选项会自动选择最佳设备和数据类型
- 如果遇到CUDA内存不足，可以尝试将device设置为"cpu"

- It is recommended to use this node immediately after loading the Anima model
- The "auto" option automatically selects the best device and data type
- If you encounter CUDA out of memory, try setting device to "cpu"

---

### 3. Anima Style Embeds From CLIP Vision
**功能描述 | Function Description**  
从CLIP Vision的输出中提取风格嵌入向量，用于支持风格参考的Anima Adapter。

Extracts style embedding vectors from CLIP Vision output for use with style-reference Anima Adapters.

**输入参数 | Input Parameters**
| 参数名 | Parameter Name | 类型 | Type | 默认值 | Default Value | 取值范围 | Value Range | 说明 | Description |
|--------|----------------|------|------|--------|---------------|----------|-------------|------|-------------|
| clip_vision_output | clip_vision_output | CLIP_VISION_OUTPUT | CLIP_VISION_OUTPUT | - | CLIP Vision节点的输出 | Output from CLIP Vision node | 包含参考图像特征的CLIP Vision输出 | CLIP Vision output containing reference image features |
| target_dim | target_dim | 整数 | Integer | 768 | 1 ~ 4096 | 目标嵌入维度，需要与Adapter要求的维度匹配 | Target embedding dimension, needs to match the dimension required by the Adapter |
| normalize | normalize | 布尔值 | Boolean | False | True/False | 是否对嵌入向量进行归一化 | Whether to normalize the embedding vector |

**输出 | Output**
- `style_embeds`: 提取的风格嵌入向量，可以传递给模型烧录器
- `style_embeds`: Extracted style embedding vector that can be passed to the model burner

**使用注意事项 | Usage Notes**
- 大多数Anima风格Adapter使用768维的嵌入向量
- 建议使用与Anima模型配套的CLIP Vision模型
- 参考图像的质量和内容会直接影响风格迁移效果

- Most Anima style Adapters use 768-dimensional embedding vectors
- It is recommended to use the CLIP Vision model that matches the Anima model
- The quality and content of the reference image will directly affect the style transfer effect

---

### 4. Anima模型烧录器 | Anima Model Burner (SeparateModelMixerDictFuser)
**功能描述 | Function Description**  
将基础模型、VAE、CLIP、LoRA和Anima Adapter合并为一个完整的模型，永久烧录所有效果，生成可以直接使用的单文件模型。

Merges the base model, VAE, CLIP, LoRA, and Anima Adapter into a complete model, permanently baking all effects to generate a single-file model that can be used directly.

**输入参数 | Input Parameters**
| 参数名 | Parameter Name | 类型 | Type | 默认值 | Default Value | 取值范围 | Value Range | 说明 | Description |
|--------|----------------|------|------|--------|---------------|----------|-------------|------|-------------|
| model | model | 下拉菜单 | Dropdown | - | 所有可用的Anima基础模型 | All available Anima base models | Anima基础模型文件 | Anima base model file |
| clip | clip | 下拉菜单 | Dropdown | - | 所有可用的CLIP模型 | All available CLIP models | Anima配套的Qwen3 CLIP模型 | Qwen3 CLIP model for Anima |
| vae | vae | 下拉菜单 | Dropdown | - | 所有可用的VAE模型 | All available VAE models | Anima配套的VAE模型 | VAE model for Anima |
| lora_stack | lora_stack | LORA_STACK | LORA_STACK | None | LoRA堆节点的输出 | Output from LoRA stack node | 可选，要烧录的LoRA效果 | Optional, LoRA effects to bake |
| anima_adapter_stack | anima_adapter_stack | ANIMA_ADAPTER_STACK | ANIMA_ADAPTER_STACK | None | Anima Adapter堆节点的输出 | Output from Anima Adapter stack node | 可选，要烧录的Adapter效果 | Optional, Adapter effects to bake |
| style_embeds | style_embeds | ANIMA_STYLE_EMBEDS | ANIMA_STYLE_EMBEDS | None | 风格嵌入节点的输出 | Output from style embedding node | 可选，风格参考嵌入向量 | Optional, style reference embedding vector |
| save_dtype | save_dtype | 下拉菜单 | Dropdown | auto | auto, float16, bfloat16, float32 | 烧录后模型的保存数据类型 | Data type for the baked model |
| device | device | 下拉菜单 | Dropdown | auto | auto, cpu, cuda, npu | 烧录过程使用的设备 | Device used for the baking process |
| adapter_device | adapter_device | 下拉菜单 | Dropdown | auto | auto, cpu, cuda, npu | Adapter运行使用的设备 | Device used for Adapter execution |
| adapter_dtype | adapter_dtype | 下拉菜单 | Dropdown | auto | auto, from_file, float16, bfloat16, float32 | Adapter运行使用的数据类型 | Data type used for Adapter execution |
| adapter_num_blocks | adapter_num_blocks | 整数 | Integer | 28 | 1 ~ 128 | Anima DiT的块数量，通常为28 | Number of blocks in Anima DiT, usually 28 |

**输出 | Output**
- `MODEL`: 烧录后的完整模型
- `CLIP`: 烧录后的CLIP模型
- `VAE`: 烧录后的VAE模型
- `MODEL`: Complete baked model
- `CLIP`: Baked CLIP model
- `VAE`: Baked VAE model

**使用注意事项 | Usage Notes**
- 烧录过程会消耗大量显存，建议使用CUDA设备
- "auto"选项会自动选择最佳的设备和数据类型
- 烧录后的模型可以直接保存为safetensors文件，无需再次加载LoRA和Adapter
- 如果烧录过程中出现显存不足，可以尝试将device设置为"cpu"，但速度会变慢

- The baking process consumes a lot of VRAM, it is recommended to use a CUDA device
- The "auto" option automatically selects the best device and data type
- The baked model can be directly saved as a safetensors file without loading LoRA and Adapter again
- If you encounter out of memory during baking, try setting device to "cpu", but it will be slower

## 快速开始指南 | Quick Start Guide

### 基本使用流程 | Basic Usage Flow
1. 加载Anima基础模型
2. 使用`Anima Adapter 插件堆`节点选择要应用的Adapter
3. 将Adapter堆连接到`Anima模型烧录器`的`anima_adapter_stack`输入
4. 选择对应的CLIP和VAE模型
5. 运行工作流，生成烧录后的模型
6. 使用烧录后的模型进行图像生成

1. Load the Anima base model
2. Use the `Anima Adapter Stack` node to select the Adapter to apply
3. Connect the Adapter stack to the `anima_adapter_stack` input of the `Anima Model Burner`
4. Select the corresponding CLIP and VAE models
5. Run the workflow to generate the baked model
6. Use the baked model for image generation

### 风格参考工作流 | Style Reference Workflow
1. 使用`CLIP Vision Loader`节点加载CLIP Vision模型
2. 使用`Load Image`节点加载参考风格图像
3. 将图像连接到`CLIP Vision Encode`节点
4. 将CLIP Vision输出连接到`Anima Style Embeds From CLIP Vision`节点
5. 将生成的style_embeds连接到`Anima模型烧录器`的`style_embeds`输入
6. 选择支持风格参考的Adapter并烧录模型

1. Use the `CLIP Vision Loader` node to load the CLIP Vision model
2. Use the `Load Image` node to load the reference style image
3. Connect the image to the `CLIP Vision Encode` node
4. Connect the CLIP Vision output to the `Anima Style Embeds From CLIP Vision` node
5. Connect the generated style_embeds to the `style_embeds` input of the `Anima Model Burner`
6. Select a style-reference compatible Adapter and bake the model

### 多效果叠加工作流 | Multi-Effect Stacking Workflow
1. 创建多个`Anima Adapter 插件堆`节点
2. 将第一个Adapter堆的输出连接到第二个Adapter堆的`prev_stack`输入
3. 以此类推，最多可以叠加任意数量的Adapter
4. 将最终的Adapter堆连接到模型烧录器
5. 同时可以添加LoRA堆，实现LoRA+Adapter的混合效果

1. Create multiple `Anima Adapter Stack` nodes
2. Connect the output of the first Adapter stack to the `prev_stack` input of the second Adapter stack
3. And so on, you can stack any number of Adapters
4. Connect the final Adapter stack to the model burner
5. You can also add a LoRA stack to achieve mixed LoRA+Adapter effects

## 常见问题与故障排除 | FAQ & Troubleshooting

### Q: 加载模型时出现"tensor on different devices"错误怎么办？
### Q: What should I do if I get a "tensor on different devices" error when loading the model?
A: 在加载模型后立即使用`Anima 设备修复器`节点，将模型连接到该节点的输入，然后使用修复后的模型进行后续操作。

A: Use the `Anima Device Fix` node immediately after loading the model, connect the model to the input of this node, and then use the fixed model for subsequent operations.

### Q: Adapter应用后没有效果怎么办？
### Q: What should I do if the Adapter has no effect after application?
A: 
1. 检查Adapter的强度是否大于0
2. 确认Adapter文件是否正确放入了`models/anima_adapters`目录
3. 刷新ComfyUI网页界面
4. 尝试降低或提高Adapter的强度
5. 确认使用的是Anima DiT模型，其他模型不支持此Adapter

A:
1. Check if the Adapter strength is greater than 0
2. Confirm that the Adapter file is correctly placed in the `models/anima_adapters` directory
3. Refresh the ComfyUI web interface
4. Try lowering or increasing the Adapter strength
5. Confirm that you are using an Anima DiT model, other models do not support this Adapter

### Q: 烧录模型时出现显存不足怎么办？
### Q: What should I do if I get out of memory when baking the model?
A:
1. 将`device`参数设置为"cpu"，使用CPU进行烧录（速度较慢）
2. 关闭其他占用显存的程序
3. 减少同时烧录的LoRA和Adapter数量
4. 使用`float16`或`bfloat16`数据类型

A:
1. Set the `device` parameter to "cpu" to use CPU for baking (slower)
2. Close other programs that occupy VRAM
3. Reduce the number of LoRAs and Adapters being baked at the same time
4. Use `float16` or `bfloat16` data type

### Q: 风格参考效果不好怎么办？
### Q: What should I do if the style reference effect is not good?
A:
1. 使用高质量的参考图像
2. 确保参考图像的风格特征明显
3. 尝试调整`normalize`参数
4. 确认`target_dim`与Adapter要求的维度一致
5. 尝试不同的Adapter强度

A:
1. Use high-quality reference images
2. Ensure that the reference image has obvious style features
3. Try adjusting the `normalize` parameter
4. Confirm that `target_dim` matches the dimension required by the Adapter
5. Try different Adapter strengths

## 注意事项 | Notes
- 本插件仅支持Anima DiT系列模型，其他模型无法使用
- Adapter文件通常以`.safetensors`或`.pt`格式提供
- 烧录后的模型可以在任何支持Anima模型的ComfyUI环境中使用
- 建议定期更新插件以获得最新功能和修复

- This plugin only supports Anima DiT series models, other models cannot be used
- Adapter files are usually provided in `.safetensors` or `.pt` format
- Baked models can be used in any ComfyUI environment that supports Anima models
- It is recommended to update the plugin regularly to get the latest features and fixes
