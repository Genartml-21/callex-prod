"""
Callex TTS version tracking.
Follows Semantic Versioning: MAJOR.MINOR.PATCH
"""

__version__ = "2.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Model compatibility version — increment when checkpoint format changes
__model_version__ = "2.0"
