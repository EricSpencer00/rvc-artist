#!/usr/bin/env python3
"""
Download Yeat vocal samples from YouTube for training dataset.
Uses yt-dlp to download high-quality audio, then separates vocals using Demucs.
"""

import os
import subprocess
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from datetime import datetime

# Yeat songs with strong vocal presence - URLs would go here
YEAT_SONGS = [
    {
        "title": "Luh Geek",
        "url_placeholder": "https://www.youtube.com/watch?v=...",
        "duration_est": 180
    },
    {
        "title": "Kant Dïe", 
        "url_placeholder": "https://www.youtube.com/watch?v=...",
        "duration_est": 210
    },
    {
        "title": "Luv Money",
        "url_placeholder": "https://www.youtube.com/watch?v=...",
        "duration_est": 195
    },
    {
        "title": "Poppin",
        "url_placeholder": "https://www.youtube.com/watch?v=...",
        "duration_est": 165
    },
    {
        "title": "Hatër",
        "url_placeholder": "https://www.youtube.com/watch?v=...",
        "duration_est": 180
    },
]

def setup_directories():
    """Create necessary directories."""
    dirs = [
        "data/training/yeat_raw",
        "data/training/yeat_separated",
        "data/training/yeat_vocals",
        "data/training/yeat_instrumentals"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs

def download_audio(url, output_path):
    """Download audio from YouTube using yt-dlp."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "192",
            "-o", str(output_path),
            url
        ]
        print(f"  Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Downloaded to {output_path}")
            return True
        else:
            print(f"  ❌ Download failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("  ❌ yt-dlp not installed. Install with: brew install yt-dlp")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def separate_stems(audio_path, output_dir):
    """Separate vocals and instrumental using Demucs."""
    try:
        from demucs.separate import separate
        from demucs.pretrained import get_model
        
        print(f"  Separating stems from {audio_path}...")
        
        # Load audio
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # Convert stereo to mono if needed
        
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        # Resample to 44100 if needed (Demucs standard)
        if sr != 44100:
            from scipy import signal
            num_samples = int(len(audio) * 44100 / sr)
            audio = signal.resample(audio, num_samples)
            sr = 44100
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        else:
            audio = audio.T
        
        # Get Demucs model
        model = get_model("htdemucs")
        
        # Separate
        stems = separate(model, torch.from_numpy(audio[np.newaxis, :, :]).float())
        
        # Extract vocals and instrumental
        vocals = stems[3].numpy()[0]  # vocals channel
        instrumental = (stems[0].numpy()[0] + stems[1].numpy()[0] + stems[2].numpy()[0]) / 3  # drums, bass, other
        
        # Save separated stems
        vocal_path = Path(output_dir) / f"{Path(audio_path).stem}_vocal.wav"
        instrumental_path = Path(output_dir) / f"{Path(audio_path).stem}_instrumental.wav"
        
        sf.write(str(vocal_path), vocals.T, sr)
        sf.write(str(instrumental_path), instrumental.T, sr)
        
        print(f"  ✅ Vocal: {vocal_path}")
        print(f"  ✅ Instrumental: {instrumental_path}")
        
        return vocal_path, instrumental_path
        
    except ImportError:
        print("  ⚠️  Demucs not available. Install with: pip install demucs julius julius julius julius")
        return None, None
    except Exception as e:
        print(f"  ❌ Separation failed: {e}")
        return None, None

def validate_vocal_sample(audio_path, min_duration=10):
    """Check if vocal sample is valid for training."""
    try:
        info = sf.info(str(audio_path))
        duration = info.duration
        
        if duration < min_duration:
            print(f"  ⚠️  Sample too short ({duration:.1f}s < {min_duration}s)")
            return False
        
        # Check for actual audio content
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            print(f"  ⚠️  Sample too quiet (RMS: {rms:.4f})")
            return False
        
        print(f"  ✅ Valid sample: {duration:.1f}s, RMS: {rms:.4f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False

def organize_dataset():
    """Organize collected vocals into training dataset."""
    vocal_dir = Path("data/training/yeat_vocals")
    if not vocal_dir.exists():
        print("No vocal samples found")
        return
    
    samples = list(vocal_dir.glob("*_vocal.wav"))
    print(f"\n📊 Dataset Summary:")
    print(f"  Total vocal samples: {len(samples)}")
    
    total_duration = 0
    for sample in samples:
        info = sf.info(str(sample))
        total_duration += info.duration
        print(f"    {sample.name}: {info.duration:.1f}s")
    
    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"\n  Training set ready at: {vocal_dir}")
    
    # Create metadata file
    metadata = {
        "collected_at": datetime.now().isoformat(),
        "num_samples": len(samples),
        "total_duration_seconds": total_duration,
        "samples": [str(s.name) for s in samples]
    }
    
    with open(Path("data/training/yeat_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to data/training/yeat_metadata.json")

def main():
    """Main collection workflow."""
    print("=" * 60)
    print("🎤 YEAT VOCAL SAMPLE COLLECTION")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    print("\n⚠️  IMPORTANT: Before running this script:")
    print("  1. Install dependencies:")
    print("     brew install yt-dlp")
    print("     pip install demucs julius")
    print("  2. Update YEAT_SONGS with actual YouTube URLs")
    print("  3. Ensure you have rights to download/use audio")
    
    print("\n📥 Download Workflow:")
    print("  1. Download high-quality audio from YouTube")
    print("  2. Separate vocals from instrumental using Demucs")
    print("  3. Validate and organize samples")
    print("  4. Prepare training dataset")
    
    print("\n🔄 Processing songs...")
    
    valid_samples = 0
    for song in YEAT_SONGS:
        print(f"\n🎵 {song['title']}")
        print(f"  (Replace URL placeholder with actual link)")
        
        # In production, replace url_placeholder with real URLs
        if "placeholder" in song["url_placeholder"].lower():
            print(f"  ⏭️  Skipping (URL placeholder)")
            continue
        
        # Download
        output_file = Path("data/training/yeat_raw") / f"{song['title']}.wav"
        if not download_audio(song["url_placeholder"], output_file):
            continue
        
        # Separate
        vocal_path, instrumental_path = separate_stems(
            output_file, 
            "data/training/yeat_separated"
        )
        
        if vocal_path and validate_vocal_sample(vocal_path):
            # Copy to organized folder
            final_path = Path("data/training/yeat_vocals") / vocal_path.name
            if vocal_path.exists():
                import shutil
                shutil.copy(str(vocal_path), str(final_path))
            valid_samples += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Collection complete!")
    print(f"  Valid samples: {valid_samples}")
    
    # Organize and summarize
    organize_dataset()
    
    if valid_samples >= 10:
        print(f"\n🎯 Dataset ready for training!")
        print(f"  Next: Train RVC model with `python scripts/train_yeat_model.py`")
    else:
        print(f"\n⚠️  Need at least 10 samples for good training quality")

if __name__ == "__main__":
    main()
