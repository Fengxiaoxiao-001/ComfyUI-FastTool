from .clip_offloader import (
    VRAM_CLIP_Offloader
)

from .ModelAndLoraToModel import (
    TrueModelMixerDictFuser
)

from .LoraStack import (
    MultiLoRAStack
)

from .AnimaLoader import (
    SeparateModelMixerDictFuser
)

from .ImageObfuscation import (
    XiaoxiaoEncrypt,
    XiaoxiaoDecrypt
)


NODE_CLASS_MAPPINGS = {
    "VRAM_CLIP_Offloader": VRAM_CLIP_Offloader,
    "TrueModelMixerDictFuser": TrueModelMixerDictFuser,
    "MultiLoRAStack": MultiLoRAStack,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser,
     "XiaoxiaoEncrypt": XiaoxiaoEncrypt,
    "XiaoxiaoDecrypt": XiaoxiaoDecrypt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAM_CLIP_Offloader": "🔄 VRAM CLIP Offloader（CLIP 搬到 CPU/NPU）",
    "TrueModelMixerDictFuser": "️【SDXL】 Model Mixer ",
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器",
    "SeparateModelMixerDictFuser": "️Anima模型烧录器",
    "XiaoxiaoEncrypt": "🔒 Xiaoxiao Encrypt (潇潇图片混淆)",
    "XiaoxiaoDecrypt": "🔓 Xiaoxiao Decrypt (潇潇图片解混淆)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
