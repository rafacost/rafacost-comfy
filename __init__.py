"""
@author: rafacost
@title: DreamOmni2-GGUF
@description: Nodes to use the DreamOmni2 GGUF models in ComfyUI.
"""

from .nodes.dreamomni2 import DreamOmni2GGUF, DreamOmni2VLMLoaderGGUF, DreamOmni2ConditioningCombine, DreamOmni2MultiImageBatch, DreamOmni2ConditioningBatch



# Register the nodes
NODE_CLASS_MAPPINGS = {
    "DreamOmni2GGUF": DreamOmni2GGUF,
    "DreamOmni2VLMLoaderGGUF": DreamOmni2VLMLoaderGGUF,
    "DreamOmni2ConditioningCombine": DreamOmni2ConditioningCombine,
    "DreamOmni2MultiImageBatch": DreamOmni2MultiImageBatch,
    "DreamOmni2ConditioningBatch": DreamOmni2ConditioningBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOmni2GGUF": "DreamOmni2 (GGUF)",
    "DreamOmni2VLMLoaderGGUF": "DreamOmni2 Load VLM (GGUF)",
    "DreamOmni2ConditioningCombine": "DreamOmni2 Multi-Image Conditioning",
    "DreamOmni2MultiImageBatch": "Batch Images",
    "DreamOmni2ConditioningBatch": "DreamOmni2 Batch Conditioning",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 