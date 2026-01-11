#!/usr/bin/env python3
"""
Quick test: Generate music using the analyzed style profile
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.services.music_generator import MusicGenerator


def main():
    print("\n🎵 Generating Music from YEAT Style Profile\n")
    print("=" * 60)
    
    # Load style profile
    profile_path = Path(__file__).parent / "data" / "features" / "style_profile.json"
    
    if not profile_path.exists():
        print("❌ Style profile not found!")
        print("   Run: python test_pipeline.py first")
        return
    
    with open(profile_path) as f:
        style_profile = json.load(f)
    
    print(f"📊 Loaded style profile:")
    print(f"  • Avg Tempo: {style_profile['tempo']['mean']:.1f} BPM ({style_profile['characteristics']['tempo']})")
    print(f"  • Key: {style_profile['key']['most_common']}")
    print(f"  • Energy: {style_profile['characteristics']['energy']}")
    print(f"  • Overall: {style_profile['characteristics']['overall']}")
    
    # Generate
    print("\n🎼 Generating 30-second track...\n")
    
    generator = MusicGenerator()
    output_dir = Path(__file__).parent / "output" / "generated"
    
    try:
        output_file = generator.generate(
            prompt="energetic trap beat in the style of YEAT, hard 808s, fast tempo",
            duration=30,
            style_profile=style_profile,
            output_dir=str(output_dir)
        )
        
        print(f"\n✅ Success!")
        print(f"   Generated: {output_file}")
        
        # Check if it's placeholder
        if "placeholder" in output_file:
            print("\n💡 Note: This is a placeholder audio file")
            print("   For AI-generated music, install AudioCraft:")
            print("   pip install audiocraft")
            print("   (Requires ~2GB download and PyTorch)")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
