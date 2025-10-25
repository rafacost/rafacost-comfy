"""
@author: rafacost
@title: rafacostComfy Nodes
@description: Nodes to use the DreamOmni2 GGUF models in ComfyUI.
"""

from .nodes.dreamomni2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Set the web directory path relative to this __init__.py file
WEB_DIRECTORY = "./web/assets/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']