"""Services package"""

# Core services
from .youtube_downloader import YouTubeDownloader
from .style_analyzer import StyleAnalyzer
from .lyrics_generator import LyricsGenerator
from .music_generator import MusicGenerator

# Multi-stem generation system
from .multi_stem_generator import MultiStemGenerator, StemProcessor, MasteringProcessor
from .vocal_generator import VocalGenerator

__all__ = [
    'YouTubeDownloader',
    'StyleAnalyzer', 
    'LyricsGenerator',
    'MusicGenerator',
    'MultiStemGenerator',
    'StemProcessor',
    'MasteringProcessor',
    'VocalGenerator',
]
