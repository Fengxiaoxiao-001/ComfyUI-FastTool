from .clip_offloader import (
    VRAM_CLIP_Offloader
)

from .ModelAndLoraToModel import (
    TrueModelMixerDictFuser
)

from .LoraStack import (
    MultiLoRAStack
)

NODE_CLASS_MAPPINGS = {
    "VRAM_CLIP_Offloader": VRAM_CLIP_Offloader,
    "TrueModelMixerDictFuser": TrueModelMixerDictFuser,
    "MultiLoRAStack": MultiLoRAStack,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser
}

from .AnimaLoader import (
    SeparateModelMixerDictFuser
)


NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAM_CLIP_Offloader": "🔄 VRAM CLIP Offloader（CLIP 搬到 CPU/NPU）",
    "TrueModelMixerDictFuser": "️【SDXL】 Model Mixer ",
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器",
    "SeparateModelMixerDictFuser": "️Anima模型烧录器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
