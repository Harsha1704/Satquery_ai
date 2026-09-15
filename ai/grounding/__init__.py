# ai/grounding/__init__.py

# from .grounder import TextGuidedGrounder
from .remoteclip_grounder import RemoteCLIPGrounder

__all__ = [
    # "TextGuidedGrounder",
    "RemoteCLIPGrounder",
]