"""
RVC Artist - AI Music Generation Pipeline
==========================================
Creates AI-generated songs by analyzing an artist's existing catalog.

Components:
1. YouTube Downloader - Downloads audio from playlists
2. Audio Transcriber - Uses Whisper for vocal transcription
3. Lyrics Scraper - Fetches lyrics from Genius
4. Lyrics Aligner - Matches transcriptions with official lyrics
5. Style Analyzer - Extracts musical features and patterns
6. Song Generator - Creates new songs using fine-tuned models
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Create necessary directories
DIRS = [
    "./data",
    "./data/audio",
    "./data/audio/raw",
    "./data/audio/vocals",
    "./data/audio/instrumentals",
    "./data/transcripts",
    "./data/lyrics",
    "./data/aligned",
    "./data/features",
    "./models",
    "./models/checkpoints",
    "./output",
    "./output/generated",
    "./logs"
]

for dir_path in DIRS:
    os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    from src.app import create_app
    
    app = create_app()
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    RVC Artist Studio                          ║
    ║              AI Music Generation Pipeline                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Server running at: http://{host}:{port}                       ║
    ║  Debug mode: {debug}                                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug)
