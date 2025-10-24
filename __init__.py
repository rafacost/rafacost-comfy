"""
@author: rafacost
@title: DreamOmni2-GGUF
@description: Nodes to use the DreamOmni2 GGUF models in ComfyUI.
"""

from .nodes.dreamomni2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']