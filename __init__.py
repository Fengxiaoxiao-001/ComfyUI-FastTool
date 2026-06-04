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
    AnimaAdapterStack,
    AnimaDeviceFix,
    AnimaStyleEmbedsFromClipVision,
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
    "AnimaAdapterStack": AnimaAdapterStack,
    "AnimaStyleEmbedsFromClipVision": AnimaStyleEmbedsFromClipVision,
    "AnimaDeviceFix": AnimaDeviceFix,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser,
    "XiaoxiaoEncrypt": XiaoxiaoEncrypt,
    "XiaoxiaoDecrypt": XiaoxiaoDecrypt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAM_CLIP_Offloader": "🔄 VRAM CLIP Offloader（CLIP 搬到 CPU/NPU）",
    "TrueModelMixerDictFuser": "️【SDXL】 Model Mixer ",
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器",
    "AnimaAdapterStack": "Anima Adapter 插件堆",
    "AnimaStyleEmbedsFromClipVision": "Anima Style Embeds From CLIP Vision",
    "AnimaDeviceFix": "Anima 设备修复器",
    "SeparateModelMixerDictFuser": "️Anima模型烧录器",
    "XiaoxiaoEncrypt": "🔒 Xiaoxiao Encrypt (潇潇图片混淆)",
    "XiaoxiaoDecrypt": "🔓 Xiaoxiao Decrypt (潇潇图片解混淆)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
