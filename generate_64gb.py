#!/usr/bin/env python3
"""
Quick start script to generate music using 64GB RAM optimizations.
"""

from config_64gb import get_config, CONFIG_64GB_OPTIMAL
from src.services.multi_stem_generator import MultiStemGenerator


def main():
    """Generate music using optimized 64GB config."""
    
    # Get optimal config for your machine
    config = get_config('64gb_optimal')
    
    print("\n" + "=" * 70)
    print("🚀 MULTI-STEM MUSIC GENERATOR - 64GB RAM OPTIMIZED")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   • Model Size: {config.model_size}")
    print(f"   • Parallel Stems: {config.max_parallel_stems} at a time")
    print(f"   • Model Caching: {'✅' if config.cache_all_models else '❌'}")
    print(f"   • Processing: {'✅' if config.enable_processing else '❌'}")
    print(f"   • Mastering: {'✅' if config.enable_mastering else '❌'}")
    print("=" * 70)
    
    # Initialize generator with optimizations
    print("\n📦 Loading models...")
    generator = MultiStemGenerator(**config.to_dict())
    
    # Example generation
    prompt = "aggressive trap beat, heavy 808s, dark atmosphere, emotional vibes"
    
    print(f"\n🎵 Generating song from prompt:")
    print(f"   \"{prompt}\"")
    print(f"\n⏳ This uses parallel generation for faster output...")
    
    mixed_path, stem_paths = generator.generate(
        prompt=prompt,
        duration=15,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1', 'melody_2'],
        output_dir="./output/generated",
        save_individual_stems=True,
        temperature=config.default_temperature,
        guidance_scale=config.default_guidance_scale
    )
    
    print(f"\n{'='*70}")
    print("✅ GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output Files:")
    print(f"   Mixed: {mixed_path}")
    print(f"\n🎚️  Individual Stems:")
    for stem_type, path in stem_paths.items():
        print(f"   • {stem_type}: {path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Generation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
