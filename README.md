# 🚀 ComfyUI-FastTool

**Alleviate VRAM pressure, improve K-sampling speed.**
**缓解显存压力，显著提升 K 采样速度。**

## 📖 Introduction | 简介

**[EN]**
A professional model fusion and VRAM optimization toolkit for ComfyUI. **ComfyUI-FastTool** efficiently bakes LoRA into base models with customizable precision (BF16/FP16/FP8). By offloading CLIP weights to system memory (RAM), it drastically reduces VRAM overhead and accelerates K-sampling speed. It is a must-have tool for users with limited VRAM (e.g., 8GB/12GB GPUs) pushing for high-resolution generations.

**[CN]**
一款专业的 ComfyUI 模型融合与显存优化工具箱。**ComfyUI-FastTool** 能够将 LoRA 彻底物理烧录到底模中（支持 BF16/FP16/FP8 精度自定义），并支持将 CLIP 权重移出显存至系统内存（RAM），从而大幅释放显存占用，显著提升 K 采样性能。对于追求高分辨率出图但受限于显存（如 8G/12G 显卡）的用户来说，这是必备的优化神器。

---

## ✨ Included Nodes | 包含节点

This toolkit provides three core nodes to streamline your workflow:
本工具箱提供三个核心节点来优化你的工作流：

1. 🛠️ **True Model Mixer (Base Model Loader / 底模加载节点)**
* **[EN]** Fuses and bakes LoRA weights directly into the UNET with high precision. Prevents runtime calculation overhead during sampling.
* **[CN]** 将 LoRA 权重高精度直接烧录到 UNET 底模中，避免采样时的实时计算开销。


2. 📋 **Multi LoRA Stack (LoRA Stack Loader / LoRA堆加载节点)**
* **[EN]** Easily manage, adjust, and stack multiple LoRAs before baking.
* **[CN]** 便捷地管理、调整和堆叠多个 LoRA，为物理烧录做准备。


3. 🔄 **VRAM CLIP Offloader (CLIP Transfer Node / CLIP 转移节点)**
* **[EN]** Intercepts the CLIP model and forces it into system RAM, freeing up precious VRAM for the K-Sampler.
* **[CN]** 拦截 CLIP 模型并强制将其卸载至系统内存（RAM），为 K 采样器腾出宝贵的显存空间。



---

## 📌 How to use the CLIP Offloader | 如何正确使用 CLIP 转移节点

**⚠️ CRITICAL / 极度重要:**
To ensure proper VRAM optimization without breaking prompt encoding, the `VRAM_CLIP_Offloader` must be placed **AFTER** all model/LoRA loading, but **BEFORE** the CLIP Text Encode (Prompt) nodes.

为了确保显存优化生效且不破坏提示词编码，`VRAM_CLIP_Offloader` 必须放置在所有底模/LoRA 加载**之后**，并且在 CLIP 文本编码节点**之前**。

### 🧩 Workflow Placement / 连线示意图

```text
  [Checkpoint Loader] (Outputs CLIP)
           │
           ▼
  [LoRA Loader / LoRA Stack] (Passes CLIP through)
           │
           ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   🔄 VRAM_CLIP_Offloader            ┃  <--- MUST BE PLACED HERE! (必须接在这里!)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           ▼
  [CLIP Text Encode (Prompt)] 

```

---

## ⚙️ Installation | 安装方法

**Method 1: ComfyUI Manager (Recommended)**

1. Open ComfyUI Manager.
2. Search for `FastTool` or `True Model Mixer`.
3. Click "Install" and restart ComfyUI.

**方法 1：通过 ComfyUI Manager（推荐）**

1. 打开 ComfyUI Manager 界面。
2. 搜索 `FastTool` 或 `True Model Mixer`。
3. 点击安装（Install），完成后重启 ComfyUI。

**Method 2: Manual Git Clone**
**方法 2：手动 Git 克隆**
Navigate to your ComfyUI `custom_nodes` folder and run:
前往你的 ComfyUI `custom_nodes` 文件夹，运行以下命令：

```bash
cd custom_nodes
git clone https://github.com/Fengxiaoxiao-001/ComfyUI-FastTool.git

```

Restart ComfyUI. (重启 ComfyUI)
