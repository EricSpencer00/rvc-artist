#!/usr/bin/env python3
"""
Quick Example: Multi-Stem Generation with Vocals
=================================================
Demonstrates the new multi-stem generation system.
"""

from pathlib import Path
from src.services.multi_stem_generator import MultiStemGenerator
from src.services.lyrics_generator import LyricsGenerator
from src.services.style_analyzer import StyleAnalyzer


def example_basic():
    """Basic multi-stem generation (no vocals)."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Multi-Stem (Drums + Bass + Melody)")
    print("="*60)
    
    generator = MultiStemGenerator(model_size="large")
    
    mixed_file, stems = generator.generate(
        prompt="aggressive trap beat, heavy 808 bass, dark synths, rage trap style",
        duration=10,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1'],
        save_individual_stems=True,
        guidance_scale=3.5
    )
    
    print(f"\n✅ Generated: {mixed_file}")
    print(f"   Stems: {list(stems.keys())}")


def example_with_lyrics():
    """Multi-stem generation with generated lyrics and vocals."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complete Song with Vocals")
    print("="*60)
    
    # Step 1: Generate lyrics
    print("\n1. Generating lyrics...")
    lyrics_gen = LyricsGenerator()
    
    # Load existing transcripts if available
    transcripts_dir = Path("data/transcripts")
    if transcripts_dir.exists():
        transcript_files = list(transcripts_dir.glob("*.json"))
        if transcript_files:
            lyrics_gen.load_corpus([str(f) for f in transcript_files[:5]])
    
    # Generate song structure
    lyrics = lyrics_gen.generate_full_song(
        structure=['verse', 'chorus', 'verse', 'chorus'],
        rhyme_scheme='AABB'
    )
    
    print(f"Generated lyrics:")
    print("-" * 40)
    print(lyrics[:300] + "..." if len(lyrics) > 300 else lyrics)
    print("-" * 40)
    
    # Step 2: Generate music with vocals
    print("\n2. Generating multi-stem music with vocals...")
    
    generator = MultiStemGenerator(
        model_size="large",
        vocal_model="bark"  # Use Bark for vocals (requires: pip install bark)
    )
    
    mixed_file, stems = generator.generate(
        prompt="aggressive trap beat, dark atmosphere, 808 bass, rage trap",
        duration=15,
        artist_name="Yeat",
        lyrics=lyrics,
        stems_to_generate=['vocals', 'drums', 'bass', 'melody_1'],
        save_individual_stems=True,
        temperature=1.0,
        guidance_scale=3.5
    )
    
    print(f"\n✅ Complete song generated!")
    print(f"   Mixed file: {mixed_file}")
    print(f"   Individual stems:")
    for stem_type, path in stems.items():
        print(f"      - {stem_type}: {Path(path).name}")


def example_style_guided():
    """Generate using analyzed style profile."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Style Profile-Guided Generation")
    print("="*60)
    
    # Check if we have a style profile
    features_dir = Path("data/features")
    profile_file = features_dir / "style_profile.json"
    
    if not profile_file.exists():
        print("⚠️  No style profile found. Run test_pipeline.py first to create one.")
        print("   Or analyze some songs manually:")
        print("   ```")
        print("   from src.services.style_analyzer import StyleAnalyzer")
        print("   analyzer = StyleAnalyzer()")
        print("   profile = analyzer.analyze_subset(['song1.mp3', 'song2.mp3'])")
        print("   ```")
        return
    
    # Load style profile
    import json
    with open(profile_file, 'r') as f:
        style_profile = json.load(f)
    
    print(f"\n📊 Loaded style profile:")
    print(f"   Songs analyzed: {style_profile.get('num_songs_analyzed', 0)}")
    print(f"   Avg tempo: {style_profile.get('tempo', {}).get('mean', 0):.1f} BPM")
    print(f"   Key: {style_profile.get('key', {}).get('most_common', 'unknown')}")
    
    # Build prompt from profile
    analyzer = StyleAnalyzer()
    descriptors = analyzer.features_to_prompt_descriptors(style_profile)
    prompt = ", ".join(descriptors[:8])
    
    print(f"\n   Generated prompt: {prompt[:100]}...")
    
    # Generate
    print("\n🎵 Generating with style guidance...")
    generator = MultiStemGenerator(model_size="large")
    
    mixed_file, stems = generator.generate(
        prompt=prompt,
        duration=12,
        style_profile=style_profile,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1', 'melody_2'],
        save_individual_stems=True,
        temperature=1.0,
        guidance_scale=4.0  # Higher guidance for style adherence
    )
    
    print(f"\n✅ Style-guided generation complete!")
    print(f"   Output: {mixed_file}")


def example_custom_mix():
    """Generate stems and mix with custom levels."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Mixing")
    print("="*60)
    
    generator = MultiStemGenerator(model_size="large")
    
    # Generate stems
    print("\n🎵 Generating stems...")
    stems = {}
    base_prompt = "trap beat, heavy bass, dark synths"
    
    for stem_type in ['drums', 'bass', 'melody_1']:
        print(f"   Generating {stem_type}...")
        audio = generator.generate_stem(
            stem_type=stem_type,
            base_prompt=base_prompt,
            duration=8,
            temperature=1.0,
            guidance_scale=3.0
        )
        stems[stem_type] = audio
    
    # Mix with different balances
    print("\n🎚️  Creating different mixes...")
    
    mix_presets = {
        'drums_focused': {'drums': 1.0, 'bass': 0.6, 'melody_1': 0.5},
        'bass_heavy': {'drums': 0.7, 'bass': 1.0, 'melody_1': 0.6},
        'balanced': {'drums': 0.85, 'bass': 0.9, 'melody_1': 0.75}
    }
    
    import soundfile as sf
    from datetime import datetime
    output_dir = Path("output/generated")
    
    for preset_name, levels in mix_presets.items():
        mixed = generator.mix_stems(stems, mix_levels=levels)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"mix_{preset_name}_{timestamp}.wav"
        
        if mixed.ndim == 2:
            mixed_to_save = mixed.T
        else:
            mixed_to_save = mixed
        
        sf.write(output_file, mixed_to_save, 32000)
        print(f"   ✅ {preset_name}: {output_file.name}")


def main():
    """Run examples."""
    examples = [
        ("Basic Multi-Stem", example_basic),
        ("With Vocals", example_with_lyrics),
        ("Style-Guided", example_style_guided),
        ("Custom Mixing", example_custom_mix)
    ]
    
    print("\n" + "🎼"*30)
    print("MULTI-STEM GENERATION EXAMPLES")
    print("🎼"*30)
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all")
    
    try:
        choice = input(f"\nSelect example (1-{len(examples) + 1}): ").strip()
        
        if choice == str(len(examples) + 1):
            # Run all
            for name, func in examples:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Example failed: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            name, func = examples[int(choice) - 1]
            func()
        else:
            print("Invalid choice. Running basic example...")
            example_basic()
    
    except KeyboardInterrupt:
        print("\n\nExamples interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("\nNext steps:")
    print("  - Check output/generated/ for audio files")
    print("  - Try different prompts and parameters")
    print("  - Read MULTISTEM_GUIDE.md for more details")
    print("  - Read UPGRADE_GUIDE.md for migration help")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
