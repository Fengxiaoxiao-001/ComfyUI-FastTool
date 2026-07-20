# from .unload_model import (
#     VRAMModelRecorder,
#     VRAMModelCleaner,
#     VRAMFullPurge
# )
#
# from .clip_offloader import (
#     VRAM_CLIP_Offloader
# )
#
# from .XiaoXiao_Ultimate_Upscale import (
#     XiaoXiao_Ultimate_Upscale_Pro
# )

from .ModelAndLoraToModel import (
    TrueModelMixerDictFuser
)

from .LoraStack import (
    MultiLoRAStack
)

# from .AnimaLoader import (
#     # AnimaAdapterStack,
#     # AnimaDeviceFix,
#     # AnimaStyleEmbedsFromClipVision,
#     # AnimaClipVisionInspector,
#     # SeparateModelMixerDictFuser
# )

# from .anima_adapter_runtime_visual import (
#     AnimaLoadVisualAdapter,
#     AnimaApplyVisualAdapter,
#     AnimaRemoveVisualAdapter
# )

from .ImageObfuscation import (
    XiaoxiaoEncrypt,
    XiaoxiaoDecrypt
)

from .ChordEdit_SDXL import (
    SDXLChordEditNode,
    SDXLChordEditReleaseCacheNode,
)

from .AnimaMutation import (
    SeparateModelMixerDictFuser,
    MutationAnimaModelBaker
)

NODE_CLASS_MAPPINGS = {
    # "VRAMModelRecorder": VRAMModelRecorder,
    # "VRAMModelCleaner": VRAMModelCleaner,
    # "VRAMFullPurge": VRAMFullPurge,
    # "VRAM_CLIP_Offloader": VRAM_CLIP_Offloader,
    # "XiaoXiao_Ultimate_Upscale_Pro": XiaoXiao_Ultimate_Upscale_Pro,
    "TrueModelMixerDictFuser": TrueModelMixerDictFuser,
    "MultiLoRAStack": MultiLoRAStack,
    # "AnimaAdapterStack": AnimaAdapterStack,
    # "AnimaStyleEmbedsFromClipVision": AnimaStyleEmbedsFromClipVision,
    # "AnimaClipVisionInspector": AnimaClipVisionInspector,
    # "AnimaDeviceFix": AnimaDeviceFix,
    "SeparateModelMixerDictFuser": SeparateModelMixerDictFuser,
    "MutationAnimaModelBaker": MutationAnimaModelBaker,
    "XiaoxiaoEncrypt": XiaoxiaoEncrypt,
    "XiaoxiaoDecrypt": XiaoxiaoDecrypt,
    # "NcatBotManagerNode": NcatBotManagerNode
    # "AnimaLoadVisualAdapter": AnimaLoadVisualAdapter,
    # "AnimaApplyVisualAdapter": AnimaApplyVisualAdapter,
    # "AnimaRemoveVisualAdapter": AnimaRemoveVisualAdapter,
    "ComfySDXLChordEdit": SDXLChordEditNode,
    "ComfySDXLChordEditReleaseCache": (
        SDXLChordEditReleaseCacheNode
    ),
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "VRAMModelRecorder": "🔧 VRAM Model Recorder（记录模型/自动分类）",
    # "VRAMModelCleaner": "🧹 VRAM Model Cleaner（下拉选择清理）",
    # "VRAMFullPurge": "💥 VRAM Full Purge（全局一键彻底清空）",
    # "VRAM_CLIP_Offloader": "🔄 VRAM CLIP Offloader（CLIP 搬到 CPU/NPU）",
    # "XiaoXiao_Ultimate_Upscale_Pro": "XiaoXiao Ultimate Upscale Pro v7.4（颜色修复 + 超分锐化补偿 + 低锐化）",
    "TrueModelMixerDictFuser": "️ 正统 Model Mixer (字典合并版-修复版)",
    "MultiLoRAStack": "【SDXL】多 LoRA 堆叠器",
    # "AnimaAdapterStack": "Anima Adapter 插件堆",
    # "AnimaStyleEmbedsFromClipVision": "Anima Style Embeds From CLIP Vision",
    # "AnimaClipVisionInspector": "Anima CLIP Vision Inspector",
    # "AnimaDeviceFix": "Anima 设备修复器",
    "SeparateModelMixerDictFuser": "️Anima模型烧录器",
    "MutationAnimaModelBaker": "Mutation Anima变体自动烧录器",
    "XiaoxiaoEncrypt": "🔒 Xiaoxiao Encrypt (潇潇图片混淆)",
    "XiaoxiaoDecrypt": "🔓 Xiaoxiao Decrypt (潇潇图片解混淆)",
    # "NcatBotManagerNode": "QQ Bot Dynamic Manager"
    # "AnimaLoadVisualAdapter": "Anima Load Visual Adapter",
    # "AnimaApplyVisualAdapter": "Anima Apply Visual Adapter",
    # "AnimaRemoveVisualAdapter": "Anima Remove Visual Adapter",
    "ComfySDXLChordEdit": (
        "SDXL ChordEdit (MODEL/CLIP/VAE)"
    ),
    "ComfySDXLChordEditReleaseCache": (
        "SDXL ChordEdit Release Cache"
    ),
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
