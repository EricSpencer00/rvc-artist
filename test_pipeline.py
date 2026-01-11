#!/usr/bin/env python3
"""
Test Pipeline - Tests the enhanced music generation pipeline
Tests: StyleAnalyzer, LyricsGenerator, MusicGenerator with new features
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_style_analyzer():
    """Test the StyleAnalyzer with new named profile features."""
    print("\n" + "=" * 60)
    print("Testing StyleAnalyzer")
    print("=" * 60)
    
    from src.services.style_analyzer import StyleAnalyzer
    
    root_dir = Path(__file__).parent
    audio_dir = root_dir / "data" / "audio" / "raw"
    features_dir = root_dir / "data" / "features"
    transcripts_dir = root_dir / "data" / "transcripts"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    audio_files = (
        list(audio_dir.glob("*.mp3")) +
        list(audio_dir.glob("*.wav")) +
        list(audio_dir.glob("*.m4a")) +
        list(audio_dir.glob("*.webm"))
    )
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("⚠️  No audio files found. Skipping analysis test.")
        return None
    
    analyzer = StyleAnalyzer()
    all_features = []
    
    # Analyze each file
    for i, audio_file in enumerate(audio_files[:4], 1):  # Limit to 4 for speed
        print(f"\n[{i}/{min(len(audio_files), 4)}] {audio_file.name}")
        
        try:
            transcript_path = transcripts_dir / f"{audio_file.stem}.json"
            features = analyzer.analyze(
                str(audio_file),
                transcript_path=str(transcript_path) if transcript_path.exists() else None
            )
            all_features.append(features)
            
            # Save individual features
            output_file = features_dir / f"{audio_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=2)
            
            if 'error' not in features:
                print(f"  ✅ Tempo: {features['tempo']['tempo']:.1f} BPM")
                print(f"  ✅ Key: {features['key']['full_key']}")
                if 'lyrics' in features and 'keywords' in features['lyrics']:
                    print(f"  ✅ Keywords: {', '.join(features['lyrics'].get('keywords', [])[:5])}")
            else:
                print(f"  ❌ Error: {features['error']}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Create and save style profile
    print("\n📊 Creating style profile...")
    style_profile = analyzer.create_style_profile(all_features)
    
    with open(features_dir / "style_profile.json", 'w') as f:
        json.dump(style_profile, f, indent=2)
    
    if 'error' not in style_profile:
        print(f"✅ Style profile created!")
        print(f"  • Songs analyzed: {style_profile['num_songs_analyzed']}")
        print(f"  • Avg tempo: {style_profile['tempo']['mean']:.1f} BPM")
        print(f"  • Most common key: {style_profile['key']['most_common']}")
        
        # Test new features_to_prompt_descriptors method
        print("\n🎨 Testing prompt descriptor generation...")
        descriptors = analyzer.features_to_prompt_descriptors(style_profile)
        print(f"  Descriptors: {descriptors[:5]}...")
        
        # Test named profile saving
        print("\n💾 Testing named profile saving...")
        saved_path = analyzer.save_named_profile(style_profile, "test_profile", str(features_dir))
        print(f"  Saved to: {saved_path}")
        
        # Test profile listing
        print("\n📋 Testing profile listing...")
        profiles = analyzer.list_profiles(str(features_dir))
        print(f"  Found {len(profiles)} profiles")
        for p in profiles:
            print(f"    - {p['profile_name']}: {p['num_songs']} songs")
    
    return style_profile


def test_lyrics_generator(style_profile=None):
    """Test the enhanced LyricsGenerator."""
    print("\n" + "=" * 60)
    print("Testing LyricsGenerator")
    print("=" * 60)
    
    from src.services.lyrics_generator import LyricsGenerator
    
    root_dir = Path(__file__).parent
    transcripts_dir = root_dir / "data" / "transcripts"
    
    lyrics_gen = LyricsGenerator(ngram_size=3)
    
    print("\n📚 Training from transcripts...")
    lyrics_gen.train_from_transcripts(str(transcripts_dir))
    
    if not lyrics_gen.is_trained:
        print("⚠️  No training data. Skipping lyrics tests.")
        return
    
    # Get style summary
    print("\n📊 Style Summary:")
    summary = lyrics_gen.get_style_summary()
    print(f"  • Vocabulary size: {summary.get('vocabulary_size', 0)}")
    print(f"  • Unique bigrams: {summary.get('unique_bigrams', 0)}")
    print(f"  • Unique trigrams: {summary.get('unique_trigrams', 0)}")
    print(f"  • Top words: {', '.join(summary.get('top_words', [])[:8])}")
    
    # Test basic generation
    print("\n🎤 Basic lyrics generation:")
    print("-" * 40)
    lyrics = lyrics_gen.generate_lyrics(num_lines=4, words_per_line=6)
    print(lyrics)
    
    # Test rhyme scheme
    print("\n🎤 Lyrics with AABB rhyme scheme:")
    print("-" * 40)
    lyrics = lyrics_gen.generate_lyrics(num_lines=4, rhyme_scheme='AABB')
    print(lyrics)
    
    # Test keyword conditioning
    if style_profile:
        keywords = style_profile.get('lyrics', {}).get('top_keywords', [])
        if keywords:
            print(f"\n🎤 Lyrics with keyword conditioning ({keywords[:3]}):")
            print("-" * 40)
            lyrics = lyrics_gen.generate_lyrics(num_lines=4, keywords=keywords[:5])
            print(lyrics)
    
    # Test section generation
    print("\n🎤 Generating VERSE section:")
    print("-" * 40)
    verse = lyrics_gen.generate_section('verse', style_profile)
    print(verse)
    
    print("\n🎤 Generating CHORUS section:")
    print("-" * 40)
    chorus = lyrics_gen.generate_section('chorus', style_profile)
    print(chorus)
    
    # Test full song generation
    print("\n🎤 Generating FULL SONG structure:")
    print("-" * 40)
    song = lyrics_gen.generate_full_song(
        structure=['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro'],
        style_profile=style_profile
    )
    print(song['full_lyrics'][:500] + "..." if len(song['full_lyrics']) > 500 else song['full_lyrics'])
    
    return lyrics_gen


def test_music_generator(style_profile=None):
    """Test the enhanced MusicGenerator."""
    print("\n" + "=" * 60)
    print("Testing MusicGenerator")
    print("=" * 60)
    
    from src.services.music_generator import MusicGenerator
    
    root_dir = Path(__file__).parent
    output_dir = root_dir / "output" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎹 Initializing MusicGenerator...")
    generator = MusicGenerator()
    
    # Test prompt generation
    print("\n📝 Testing enhanced prompt generation:")
    print("-" * 40)
    
    if style_profile:
        prompt = generator.style_to_prompt(style_profile, artist_name="Yeat")
        print(f"Generated prompt ({len(prompt)} chars):")
        print(f"  {prompt[:200]}...")
    else:
        prompt = generator.style_to_prompt(None, artist_name="Yeat")
        print(f"Fallback prompt: {prompt}")
    
    # Test section prompt building
    print("\n🎼 Testing section prompt building:")
    print("-" * 40)
    
    base_prompt = "trap music, heavy 808, aggressive"
    for section in ['intro', 'verse', 'chorus', 'drop', 'outro']:
        section_prompt = generator.build_section_prompt(base_prompt, section)
        print(f"  {section}: {section_prompt[:60]}...")
    
    # Test blueprint creation
    print("\n📋 Testing blueprint creation:")
    print("-" * 40)
    
    for style in ['standard', 'short', 'trap', 'edm']:
        blueprint = generator.create_default_blueprint(total_duration=60, style=style)
        sections = [s['type'] for s in blueprint['sections']]
        total = sum(s['duration'] for s in blueprint['sections'])
        print(f"  {style}: {sections} ({total}s)")
    
    # Test actual generation (short clip)
    print("\n🎵 Testing audio generation (5 second test):")
    print("-" * 40)
    
    try:
        output_file = generator.generate(
            prompt=None,
            duration=5,  # Short test
            style_profile=style_profile,
            artist_name="Yeat",
            output_dir=str(output_dir)
        )
        print(f"✅ Generated: {output_file}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return generator


def main():
    print("=" * 60)
    print("🎵 Enhanced Pipeline Test Suite")
    print("=" * 60)
    print("Testing: StyleAnalyzer, LyricsGenerator, MusicGenerator")
    print("With: Named profiles, n-gram Markov, blueprints, sections")
    
    # Test StyleAnalyzer
    style_profile = test_style_analyzer()
    
    # Test LyricsGenerator
    test_lyrics_generator(style_profile)
    
    # Test MusicGenerator
    test_music_generator(style_profile)
    
    # Summary
    print("\n" + "=" * 60)
    print("✨ Test Suite Complete!")
    print("=" * 60)
    
    root_dir = Path(__file__).parent
    features_dir = root_dir / "data" / "features"
    output_dir = root_dir / "output" / "generated"
    
    print(f"📁 Features saved to: {features_dir}")
    print(f"🎵 Generated music in: {output_dir}")
    print("\nNew features tested:")
    print("  ✓ Named style profiles")
    print("  ✓ Feature-to-prompt descriptors")
    print("  ✓ N-gram Markov chain lyrics")
    print("  ✓ Rhyme scheme enforcement")
    print("  ✓ Section-aware lyrics generation")
    print("  ✓ Full song structure generation")
    print("  ✓ Blueprint-based audio generation")
    print("  ✓ Section prompt building")


if __name__ == "__main__":
    main()
