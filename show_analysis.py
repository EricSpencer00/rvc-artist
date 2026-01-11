#!/usr/bin/env python3
"""
Display the current YEAT style profile and song analyses
"""

import json
from pathlib import Path


def main():
    features_dir = Path(__file__).parent / "data" / "features"
    
    print("\n" + "=" * 70)
    print("🎵 YEAT STYLE ANALYSIS")
    print("=" * 70)
    
    # Load style profile
    profile_path = features_dir / "style_profile.json"
    if not profile_path.exists():
        print("\n❌ No style profile found. Run: python test_pipeline.py")
        return
    
    with open(profile_path) as f:
        profile = json.load(f)
    
    print("\n📊 AGGREGATED STYLE PROFILE")
    print("-" * 70)
    print(f"  Songs Analyzed: {profile['num_songs_analyzed']}")
    print(f"\n  TEMPO:")
    print(f"    Average: {profile['tempo']['mean']:.1f} BPM")
    print(f"    Range: {profile['tempo']['min']:.1f} - {profile['tempo']['max']:.1f} BPM")
    print(f"    Std Dev: {profile['tempo']['std']:.1f}")
    print(f"    Character: {profile['characteristics']['tempo']}")
    
    print(f"\n  KEY SIGNATURES:")
    for key, count in profile['key']['distribution'].items():
        print(f"    {key}: {count} song(s)")
    print(f"    Most Common: {profile['key']['most_common']}")
    
    print(f"\n  ENERGY:")
    print(f"    RMS Mean: {profile['energy']['rms_mean']:.3f}")
    print(f"    Dynamic Range: {profile['energy']['dynamic_range_mean']:.3f}")
    print(f"    Character: {profile['characteristics']['energy']}")
    
    print(f"\n  SPECTRAL:")
    print(f"    Brightness: {profile['spectral']['brightness_mean']:.1f} Hz")
    print(f"    Character: {profile['characteristics']['timbre']}")
    
    print(f"\n  STRUCTURE:")
    print(f"    Avg Duration: {profile['structure']['avg_duration']:.1f} seconds")
    
    print(f"\n  OVERALL CHARACTER:")
    print(f"    {profile['characteristics']['overall']}")
    
    # Load individual songs
    print("\n\n📀 INDIVIDUAL SONG ANALYSES")
    print("-" * 70)
    
    feature_files = sorted(features_dir.glob("*.json"))
    for i, feature_file in enumerate(feature_files, 1):
        if feature_file.name == "style_profile.json":
            continue
        
        with open(feature_file) as f:
            features = json.load(f)
        
        if 'error' in features:
            print(f"\n  [{i}] {features['audio_file']}")
            print(f"      ❌ Error: {features['error']}")
            continue
        
        print(f"\n  [{i}] {features['audio_file']}")
        print(f"      Tempo: {features['tempo']['tempo']:.1f} BPM ({features['tempo']['num_beats']} beats)")
        print(f"      Key: {features['key']['full_key']}")
        print(f"      Duration: {features['structure']['duration']:.1f}s")
        print(f"      Energy (RMS): {features['energy']['rms_mean']:.3f}")
        print(f"      Brightness: {features['spectral']['spectral_centroid_mean']:.1f} Hz")
    
    print("\n" + "=" * 70)
    print("\n💡 This data is used to guide music generation!")
    print("   Run: python generate_test.py to create new music\n")


if __name__ == "__main__":
    main()
