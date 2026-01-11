#!/usr/bin/env python3
"""
Re-analyze audio files and test generation with existing data
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.style_analyzer import StyleAnalyzer
from src.services.music_generator import MusicGenerator


def main():
    print("=" * 60)
    print("Re-analyzing Audio Files & Testing Generation")
    print("=" * 60)
    
    # Setup paths
    root_dir = Path(__file__).parent
    audio_dir = root_dir / "data" / "audio" / "raw"
    features_dir = root_dir / "data" / "features"
    output_dir = root_dir / "output" / "generated"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    audio_files = (
        list(audio_dir.glob("*.mp3")) +
        list(audio_dir.glob("*.wav")) +
        list(audio_dir.glob("*.m4a")) +
        list(audio_dir.glob("*.webm"))
    )
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    
    # Analyze each file
    print("\n🎵 Step 1: Analyzing audio files...")
    print("-" * 60)
    
    analyzer = StyleAnalyzer()
    all_features = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        
        try:
            features = analyzer.analyze(str(audio_file))
            all_features.append(features)
            
            # Save individual features
            output_file = features_dir / f"{audio_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=2)
            
            if 'error' not in features:
                print(f"  ✅ Tempo: {features['tempo']['tempo']:.1f} BPM")
                print(f"  ✅ Key: {features['key']['full_key']}")
                print(f"  ✅ Duration: {features['structure']['duration']:.1f}s")
            else:
                print(f"  ❌ Error: {features['error']}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            all_features.append({
                'audio_file': audio_file.name,
                'error': str(e)
            })
    
    # Create style profile
    print("\n\n📊 Step 2: Creating style profile...")
    print("-" * 60)
    
    style_profile = analyzer.create_style_profile(all_features)
    profile_path = features_dir / "style_profile.json"
    
    with open(profile_path, 'w') as f:
        json.dump(style_profile, f, indent=2)
    
    if 'error' not in style_profile:
        print(f"✅ Style profile created successfully!")
        print(f"  • Songs analyzed: {style_profile['num_songs_analyzed']}")
        print(f"  • Avg tempo: {style_profile['tempo']['mean']:.1f} BPM")
        print(f"  • Most common key: {style_profile['key']['most_common']}")
        print(f"  • Characteristics: {style_profile['characteristics']['overall']}")
    else:
        print(f"❌ Style profile error: {style_profile['error']}")
    
    # Test music generation
    print("\n\n🎼 Step 3: Testing music generation...")
    print("-" * 60)
    
    try:
        generator = MusicGenerator()
        
        # Generate with style profile
        print("\nGenerating 10-second sample...")
        output_file = generator.generate(
            prompt="energetic hip hop beat",
            duration=10,
            style_profile=style_profile if 'error' not in style_profile else None,
            output_dir=str(output_dir)
        )
        
        print(f"\n✅ Generation complete!")
        print(f"  Output: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    valid_analyses = len([f for f in all_features if 'error' not in f])
    print(f"✅ Successfully analyzed: {valid_analyses}/{len(audio_files)} files")
    print(f"📁 Features saved to: {features_dir}")
    print(f"🎵 Generated music in: {output_dir}")
    
    if valid_analyses == 0:
        print("\n⚠️  No files were successfully analyzed!")
        print("   Make sure librosa is installed: pip install librosa soundfile")
    elif valid_analyses < len(audio_files):
        print(f"\n⚠️  {len(audio_files) - valid_analyses} files failed analysis")
    
    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()
