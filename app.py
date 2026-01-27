#!/usr/bin/env python3
"""
RVC Artist - Multi-Stem Music Generation
==========================================
Unified CLI application for generating high-quality trap/rap music using
multi-stem synthesis with Suno v4.5+ quality features.

Usage:
    python3 app.py generate --prompt "..." --duration 30
    python3 app.py analyze --songs data/audio/
    python3 app.py download --url "https://youtube.com/watch?v=..."
    python3 app.py test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List
import os

# Suppress warnings (TensorFlow not used in this project, CUDA not available on M1 Mac)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only affects TensorFlow (not in use)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''   # No CUDA on Apple Silicon

# Create necessary directories
BASE_DIR = Path(__file__).parent
DIRS = [
    BASE_DIR / "data", BASE_DIR / "data/audio", BASE_DIR / "data/audio/raw", 
    BASE_DIR / "data/audio/vocals", BASE_DIR / "data/audio/instrumentals",
    BASE_DIR / "data/transcripts", BASE_DIR / "data/lyrics", BASE_DIR / "data/aligned",
    BASE_DIR / "data/features", BASE_DIR / "models", BASE_DIR / "models/checkpoints",
    BASE_DIR / "output", BASE_DIR / "output/generated", BASE_DIR / "logs"
]
for dir_path in DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

from src.services.multi_stem_generator import MultiStemGenerator
from src.services.style_analyzer import StyleAnalyzer
from src.services.lyrics_generator import LyricsGenerator


class RVCArtistApp:
    """Main application class."""

    def __init__(self):
        """Initialize the application."""
        self.generator = None
        self.analyzer = None
        self.lyrics_gen = None

    def _init_generator(self, use_64gb_optimizations: bool = True):
        """Initialize multi-stem generator with optimizations."""
        if self.generator is None:
            print("📦 Loading MusicGen models...")
            self.generator = MultiStemGenerator(
                model_size="large",
                enable_processing=True,
                enable_mastering=True,
                enable_parallel_generation=use_64gb_optimizations,
                max_parallel_stems=2 if use_64gb_optimizations else 1,
                cache_all_models=use_64gb_optimizations
            )
        return self.generator

    def _init_analyzer(self):
        """Initialize style analyzer."""
        if self.analyzer is None:
            print("🎵 Initializing style analyzer...")
            self.analyzer = StyleAnalyzer()
        return self.analyzer

    def _init_lyrics_gen(self):
        """Initialize lyrics generator."""
        if self.lyrics_gen is None:
            print("📚 Initializing lyrics generator...")
            self.lyrics_gen = LyricsGenerator()
        return self.lyrics_gen

    def generate(self, args) -> int:
        """Generate music from prompt."""
        print("\n" + "=" * 70)
        print("🎼 MULTI-STEM MUSIC GENERATION")
        print("=" * 70)

        generator = self._init_generator(use_64gb_optimizations=not args.no_parallel)

        # Parse stems
        stems = args.stems.split(",") if args.stems else ["drums", "bass", "melody_1"]
        stems = [s.strip() for s in stems]

        print(f"\n📋 Generation Parameters:")
        print(f"   Prompt: {args.prompt[:60]}...")
        print(f"   Duration: {args.duration}s")
        print(f"   Stems: {', '.join(stems)}")
        print(f"   Parallel: {'✅' if not args.no_parallel else '❌'}")
        print(f"   Artist: {args.artist or 'Generic'}")

        try:
            mixed_path, stem_paths = generator.generate(
                prompt=args.prompt,
                duration=args.duration,
                artist_name=args.artist,
                stems_to_generate=stems,
                output_dir=args.output_dir,
                save_individual_stems=not args.no_stems,
                temperature=args.temperature,
                guidance_scale=args.guidance
            )

            print(f"\n✅ GENERATION COMPLETE")
            print(f"   Output: {mixed_path}")
            if stem_paths:
                print(f"   Stems: {len(stem_paths)} files generated")
            print("=" * 70 + "\n")
            return 0

        except KeyboardInterrupt:
            print("\n⏸️  Generation interrupted by user.")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def analyze(self, args) -> int:
        """Analyze audio files and create style profile."""
        print("\n" + "=" * 70)
        print("🎨 STYLE ANALYSIS")
        print("=" * 70)

        analyzer = self._init_analyzer()
        audio_dir = Path(args.directory)

        if not audio_dir.exists():
            print(f"❌ Directory not found: {audio_dir}")
            return 1

        # Find audio files
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a"}
        audio_files = [
            f for f in audio_dir.rglob("*")
            if f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            return 1

        print(f"\n📁 Found {len(audio_files)} audio files")

        try:
            style_profile = analyzer.analyze_directory(str(audio_dir))

            profile_path = Path("data/features/style_profile.json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(profile_path, "w") as f:
                json.dump(style_profile, f, indent=2)

            print(f"\n✅ ANALYSIS COMPLETE")
            print(f"   Profile saved: {profile_path}")
            print(f"   Songs analyzed: {style_profile.get('num_songs_analyzed', 0)}")
            print(f"   Avg tempo: {style_profile.get('tempo', {}).get('mean', 0):.1f} BPM")
            print(f"   Key: {style_profile.get('key', {}).get('most_common', 'Unknown')}")
            print("=" * 70 + "\n")
            return 0

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def lyrics(self, args) -> int:
        """Generate lyrics based on style."""
        print("\n" + "=" * 70)
        print("📝 LYRICS GENERATION")
        print("=" * 70)

        lyrics_gen = self._init_lyrics_gen()

        # Train on transcripts if available
        transcript_dir = Path("data/transcripts")
        if transcript_dir.exists():
            transcript_files = list(transcript_dir.glob("*.json"))
            if transcript_files:
                print(f"\n📚 Training on {len(transcript_files)} transcripts...")
                lyrics_gen.train_from_transcripts(str(transcript_dir))

        try:
            print(f"\n🎤 Generating {args.section.upper()} lyrics...")
            lyrics = lyrics_gen.generate(
                section=args.section,
                length=args.length,
                rhyme_scheme=args.scheme
            )

            print(f"\n{'='*70}")
            print(lyrics)
            print("=" * 70 + "\n")
            return 0

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1

    def download(self, args) -> int:
        """Download audio from YouTube."""
        print("\n" + "=" * 70)
        print("📥 YOUTUBE DOWNLOADER")
        print("=" * 70)

        try:
            from src.services.youtube_downloader import YouTubeDownloader

            downloader = YouTubeDownloader(output_dir=args.output_dir)
            print(f"\n⏳ Downloading: {args.url}")

            audio_path = downloader.download(args.url)

            print(f"\n✅ DOWNLOAD COMPLETE")
            print(f"   Audio: {audio_path}")
            print("=" * 70 + "\n")
            return 0

        except ImportError:
            print("❌ YouTube downloader not installed")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1

    def test(self, args) -> int:
        """Run integration tests."""
        print("\n" + "=" * 70)
        print("🧪 RUNNING TESTS")
        print("=" * 70)

        try:
            import unittest
            from tests.test_battle_pipeline import BattlePipelineTests

            suite = unittest.TestLoader().loadTestsFromTestCase(BattlePipelineTests)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            print("\n" + "=" * 70)
            if result.wasSuccessful():
                print(f"✅ ALL {result.testsRun} TESTS PASSED")
                return 0
            else:
                print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
                return 1

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def info(self, args) -> int:
        """Show system and configuration info."""
        print("\n" + "=" * 70)
        print("ℹ️  SYSTEM INFORMATION")
        print("=" * 70)

        try:
            import torch
            print(f"\n🔧 PyTorch:")
            print(f"   Version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            print(f"   MPS available: {torch.backends.mps.is_available()}")
        except:
            print("\n🔧 PyTorch: Not installed")

        try:
            import audiocraft
            print(f"\n🎵 AudioCraft:")
            print(f"   Installed: ✅")
        except:
            print(f"\n🎵 AudioCraft: Not installed")

        try:
            import bark
            print(f"\n🗣️  Bark TTS:")
            print(f"   Installed: ✅")
        except:
            print(f"\n🗣️  Bark TTS: Not installed")

        try:
            import pedalboard
            print(f"\n🎚️  Pedalboard:")
            print(f"   Installed: ✅")
        except:
            print(f"\n🎚️  Pedalboard: Not installed")

        try:
            import pyloudnorm
            print(f"\n📊 Pyloudnorm:")
            print(f"   Installed: ✅")
        except:
            print(f"\n📊 Pyloudnorm: Not installed")

        print(f"\n📁 Data Directories:")
        dirs = {
            "Audio": "data/audio",
            "Features": "data/features",
            "Output": "output/generated",
            "Models": "models/checkpoints"
        }
        for name, path in dirs.items():
            p = Path(path)
            exists = "✅" if p.exists() else "❌"
            print(f"   {exists} {name}: {path}")

        print("\n" + "=" * 70 + "\n")
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RVC Artist - Multi-Stem Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate music
  %(prog)s generate --prompt "aggressive trap beat, heavy 808s"
  
  # Analyze audio directory
  %(prog)s analyze --directory data/audio
  
  # Generate lyrics
  %(prog)s lyrics --section verse --length 8
  
  # Download from YouTube
  %(prog)s download --url "https://youtube.com/watch?v=..."
  
  # Run tests
  %(prog)s test
  
  # Show system info
  %(prog)s info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate music from prompt")
    gen_parser.add_argument("--prompt", required=True, help="Music description prompt")
    gen_parser.add_argument("--duration", type=int, default=30, help="Duration in seconds (default: 30)")
    gen_parser.add_argument("--artist", help="Artist name for style reference")
    gen_parser.add_argument("--stems", default="drums,bass,melody_1", help="Stems to generate (comma-separated)")
    gen_parser.add_argument("--output-dir", default="output/generated", help="Output directory")
    gen_parser.add_argument("--no-stems", action="store_true", help="Don't save individual stems")
    gen_parser.add_argument("--no-parallel", action="store_true", help="Disable parallel generation (slower)")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    gen_parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")

    # Analyze command
    ana_parser = subparsers.add_parser("analyze", help="Analyze audio and create style profile")
    ana_parser.add_argument("--directory", default="data/audio", help="Audio directory to analyze")

    # Lyrics command
    lyr_parser = subparsers.add_parser("lyrics", help="Generate lyrics")
    lyr_parser.add_argument("--section", default="verse", choices=["intro", "verse", "chorus", "bridge", "outro"],
                           help="Song section type")
    lyr_parser.add_argument("--length", type=int, default=8, help="Lyric lines to generate")
    lyr_parser.add_argument("--scheme", default="natural", choices=["aabb", "abab", "natural"],
                           help="Rhyme scheme")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download audio from YouTube")
    dl_parser.add_argument("--url", required=True, help="YouTube URL")
    dl_parser.add_argument("--output-dir", default="data/audio", help="Output directory")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run integration tests")
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system and config info")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    app = RVCArtistApp()

    commands = {
        "generate": app.generate,
        "analyze": app.analyze,
        "lyrics": app.lyrics,
        "download": app.download,
        "test": app.test,
        "info": app.info,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    # Run the CLI application
    sys.exit(main())
    
    # Note: Flask web server code was previously here but is unreachable
    # after sys.exit(). To run a Flask server, create a separate src/app_web.py
    # or import and initialize Flask properly in a different entry point.
