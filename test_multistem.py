#!/usr/bin/env python3
"""
Test Multi-Stem Generation
===========================
Tests the new multi-stem music generator that creates separate
vocal, drum, bass, and melody tracks.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.multi_stem_generator import MultiStemGenerator
from src.services.style_analyzer import StyleAnalyzer
from src.services.lyrics_generator import LyricsGenerator


def test_basic_multistem():
    """Test basic multi-stem generation without style profile."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Multi-Stem Generation")
    print("=" * 60)
    
    generator = MultiStemGenerator(model_size="large")
    
    test_prompt = "aggressive trap beat, heavy 808s, dark atmosphere, rage trap"
    
    mixed_path, stem_paths = generator.generate(
        prompt=test_prompt,
        duration=8,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1'],
        save_individual_stems=True,
        temperature=1.0,
        guidance_scale=3.0
    )
    
    print(f"\n✅ Basic test complete!")
    print(f"   Mixed file: {mixed_path}")
    print(f"   Stems: {list(stem_paths.keys())}")


def test_full_multistem():
    """Test full multi-stem generation with all stems."""
    print("\n" + "=" * 60)
    print("Test 2: Full Multi-Stem (All Stems)")
    print("=" * 60)
    
    generator = MultiStemGenerator(model_size="large")
    
    test_prompt = "melodic trap, emotional vibes, atmospheric pads, hard-hitting drums"
    
    mixed_path, stem_paths = generator.generate(
        prompt=test_prompt,
        duration=10,
        artist_name="Playboi Carti",
        stems_to_generate=['drums', 'bass', 'melody_1', 'melody_2'],
        save_individual_stems=True,
        temperature=1.1,
        guidance_scale=4.0  # Higher guidance for more prompt adherence
    )
    
    print(f"\n✅ Full stem test complete!")
    print(f"   Mixed file: {mixed_path}")
    print(f"   Stems generated: {len(stem_paths)}")


def test_with_style_profile():
    """Test multi-stem generation with style profile guidance."""
    print("\n" + "=" * 60)
    print("Test 3: Multi-Stem with Style Profile")
    print("=" * 60)
    
    # Load existing style profile
    features_dir = Path("data/features")
    profile_file = features_dir / "style_profile.json"
    
    if not profile_file.exists():
        print("⚠️  No style profile found. Run test_pipeline.py first.")
        return
    
    with open(profile_file, 'r') as f:
        style_profile = json.load(f)
    
    print(f"📊 Loaded style profile:")
    print(f"   Songs: {style_profile.get('num_songs_analyzed', 0)}")
    print(f"   Tempo: {style_profile.get('tempo', {}).get('mean', 0):.1f} BPM")
    
    generator = MultiStemGenerator(model_size="large")
    
    # Build base prompt from style profile
    analyzer = StyleAnalyzer()
    descriptors = analyzer.features_to_prompt_descriptors(style_profile)
    base_prompt = ", ".join(descriptors[:8])
    
    print(f"   Base prompt: {base_prompt[:100]}...")
    
    mixed_path, stem_paths = generator.generate(
        prompt=base_prompt,
        duration=12,
        style_profile=style_profile,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1', 'melody_2'],
        save_individual_stems=True,
        temperature=1.0,
        guidance_scale=3.5
    )
    
    print(f"\n✅ Style-guided test complete!")
    print(f"   Mixed file: {mixed_path}")


def test_different_genres():
    """Test multi-stem generation for different genres."""
    print("\n" + "=" * 60)
    print("Test 4: Different Genre Tests")
    print("=" * 60)
    
    generator = MultiStemGenerator(model_size="large")
    
    test_cases = [
        {
            'name': 'Rage Trap',
            'prompt': 'rage trap, distorted 808s, aggressive hi-hats, dark synths',
            'artist': 'Yeat',
            'stems': ['drums', 'bass', 'melody_1']
        },
        {
            'name': 'Melodic Rap',
            'prompt': 'melodic rap, emotional piano, smooth bass, trap drums',
            'artist': 'Juice WRLD',
            'stems': ['drums', 'bass', 'melody_1', 'melody_2']
        },
        {
            'name': 'Cloud Rap',
            'prompt': 'cloud rap, ethereal pads, spacey atmosphere, minimal drums',
            'artist': 'Playboi Carti',
            'stems': ['drums', 'melody_1', 'melody_2']
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n🎵 Testing: {test['name']}")
        
        mixed_path, stem_paths = generator.generate(
            prompt=test['prompt'],
            duration=8,
            artist_name=test['artist'],
            stems_to_generate=test['stems'],
            save_individual_stems=True,
            temperature=1.0,
            guidance_scale=3.5
        )
        
        results.append({
            'genre': test['name'],
            'file': mixed_path,
            'stems': len(stem_paths)
        })
    
    print(f"\n✅ Genre tests complete!")
    for result in results:
        print(f"   {result['genre']}: {result['stems']} stems")


def test_custom_mix_levels():
    """Test custom mixing levels for different balances."""
    print("\n" + "=" * 60)
    print("Test 5: Custom Mix Levels")
    print("=" * 60)
    
    from src.services.multi_stem_generator import MultiStemGenerator
    import numpy as np
    
    generator = MultiStemGenerator(model_size="large")
    
    # Generate stems first
    print("Generating stems...")
    stems = {}
    
    for stem_type in ['drums', 'bass', 'melody_1']:
        audio = generator.generate_stem(
            stem_type=stem_type,
            base_prompt="trap beat, aggressive, dark",
            duration=8,
            temperature=1.0,
            guidance_scale=3.0
        )
        stems[stem_type] = audio
    
    # Test different mix balances
    mix_presets = {
        'drums_focused': {'drums': 1.0, 'bass': 0.7, 'melody_1': 0.5},
        'bass_heavy': {'drums': 0.7, 'bass': 1.0, 'melody_1': 0.6},
        'balanced': {'drums': 0.85, 'bass': 0.9, 'melody_1': 0.75}
    }
    
    import soundfile as sf
    output_dir = Path("output/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for preset_name, levels in mix_presets.items():
        print(f"\n🎚️  Mixing: {preset_name}")
        mixed = generator.mix_stems(stems, mix_levels=levels)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"mix_{preset_name}_{timestamp}.wav"
        
        if mixed.ndim == 2:
            mixed_to_save = mixed.T
        else:
            mixed_to_save = mixed
            
        sf.write(output_file, mixed_to_save, 32000)
        print(f"   Saved: {output_file.name}")
    
    print(f"\n✅ Mix test complete!")


def main():
    """Run all tests."""
    print("\n" + "🎼" * 30)
    print("MULTI-STEM MUSIC GENERATOR TEST SUITE")
    print("🎼" * 30)
    
    tests = [
        ("Basic Multi-Stem", test_basic_multistem),
        ("Full Multi-Stem", test_full_multistem),
        ("Style Profile Guided", test_with_style_profile),
        ("Different Genres", test_different_genres),
        ("Custom Mix Levels", test_custom_mix_levels)
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    print(f"  {len(tests) + 1}. Run all tests")
    
    try:
        choice = input(f"\nSelect test (1-{len(tests) + 1}): ").strip()
        
        if choice == str(len(tests) + 1):
            # Run all
            for name, test_func in tests:
                try:
                    test_func()
                except Exception as e:
                    print(f"❌ Test '{name}' failed: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(tests):
            # Run selected test
            name, test_func = tests[int(choice) - 1]
            test_func()
        else:
            print("Invalid choice. Running basic test...")
            test_basic_multistem()
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
