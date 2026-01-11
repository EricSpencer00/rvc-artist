# Upgrading from Single-Model to Multi-Stem Generation

## Quick Start

### 1. Install Bark for Vocals (Optional but Recommended)

```bash
# Bark is Suno's TTS model that can generate singing/rapping
pip install git+https://github.com/suno-ai/bark.git
```

Note: Bark will download ~10GB of models on first run.

### 2. Test Multi-Stem Generation

```bash
python test_multistem.py
```

Select test 1 for a quick 3-stem generation (drums, bass, melody).

### 3. Generate with Vocals

```python
from src.services.multi_stem_generator import MultiStemGenerator
from src.services.lyrics_generator import LyricsGenerator

# Generate lyrics
lyrics_gen = LyricsGenerator()
lyrics = lyrics_gen.generate_full_song()

# Generate multi-stem music with vocals
generator = MultiStemGenerator(
    model_size="large",
    vocal_model="bark"  # Use Bark for vocals
)

mixed_path, stem_paths = generator.generate(
    prompt="aggressive trap beat, heavy 808s, dark atmosphere",
    duration=15,
    artist_name="Yeat",
    lyrics=lyrics,
    stems_to_generate=['vocals', 'drums', 'bass', 'melody_1'],
    save_individual_stems=True
)

print(f"Generated: {mixed_path}")
print(f"Vocals: {stem_paths.get('vocals')}")
```

## Comparison: Before vs After

### Before (Single Model)
```python
from src.services.music_generator import MusicGenerator

generator = MusicGenerator(model_size="large")

audio_file = generator.generate(
    prompt="trap beat",
    duration=10
)

# Result: Generic instrumental beat, no vocals
```

**Issues:**
- Generic sound
- No vocals
- Can't control individual elements
- Fixed balance

### After (Multi-Stem)
```python
from src.services.multi_stem_generator import MultiStemGenerator

generator = MultiStemGenerator(
    model_size="large",
    vocal_model="bark"
)

mixed_file, stems = generator.generate(
    prompt="trap beat",
    duration=10,
    lyrics="Yeah, I'm ballin' like I'm Jordan...",
    stems_to_generate=['vocals', 'drums', 'bass', 'melody_1'],
    save_individual_stems=True
)

# Result: Proper song with vocals, balanced mix
```

**Benefits:**
- ✅ Real vocals with lyrics
- ✅ Dedicated attention to each element
- ✅ Better quality per component
- ✅ Adjustable mixing
- ✅ Can remix later using saved stems

## Feature Comparison

| Feature | Single Model | Multi-Stem |
|---------|-------------|------------|
| **Vocals** | ❌ Instrumental only | ✅ Real singing/rapping |
| **Quality** | Generic | High - focused generation |
| **Control** | Limited | Per-stem control |
| **Mix Balance** | Fixed | Fully adjustable |
| **Stems Saved** | No | Yes (optional) |
| **Remixing** | ❌ | ✅ Remix saved stems |
| **Generation Time** | 30s (1 pass) | 2-3min (5 stems) |
| **Memory Usage** | 2-4GB | 2-4GB (sequential) |

## Migration Guide

### Existing Pipeline Integration

If you have existing code using `MusicGenerator`, you can gradually migrate:

#### Step 1: Add Multi-Stem as Option

```python
# In your existing code
use_multistem = True  # Feature flag

if use_multistem:
    from src.services.multi_stem_generator import MultiStemGenerator
    generator = MultiStemGenerator(model_size="large")
    
    mixed_file, stems = generator.generate(
        prompt=prompt,
        duration=duration,
        style_profile=style_profile,
        artist_name=artist_name,
        lyrics=lyrics,
        save_individual_stems=True
    )
    audio_file = mixed_file
else:
    from src.services.music_generator import MusicGenerator
    generator = MusicGenerator(model_size="large")
    audio_file = generator.generate(
        prompt=prompt,
        duration=duration,
        style_profile=style_profile
    )
```

#### Step 2: Update Routes (Flask API)

```python
# In src/routes/pipeline.py or api.py

@app.route('/generate', methods=['POST'])
def generate_music():
    data = request.json
    
    # Extract parameters
    prompt = data.get('prompt', '')
    duration = data.get('duration', 15)
    use_multistem = data.get('multistem', True)
    include_vocals = data.get('vocals', True)
    lyrics = data.get('lyrics', None)
    
    if use_multistem:
        generator = MultiStemGenerator(
            model_size="large",
            vocal_model="bark" if include_vocals else None
        )
        
        stems_to_gen = ['drums', 'bass', 'melody_1']
        if include_vocals and lyrics:
            stems_to_gen.insert(0, 'vocals')
        
        mixed_file, stem_files = generator.generate(
            prompt=prompt,
            duration=duration,
            lyrics=lyrics,
            stems_to_generate=stems_to_gen,
            save_individual_stems=True
        )
        
        return {
            'audio_file': mixed_file,
            'stems': stem_files,
            'method': 'multistem'
        }
    else:
        # Fallback to single model
        generator = MusicGenerator(model_size="large")
        audio_file = generator.generate(
            prompt=prompt,
            duration=duration
        )
        return {
            'audio_file': audio_file,
            'method': 'single'
        }
```

#### Step 3: Update Frontend

```html
<!-- In frontend/templates/index.html -->

<div class="generation-options">
    <label>
        <input type="checkbox" id="useMultistem" checked>
        Use Multi-Stem Generation (Better Quality)
    </label>
    
    <label>
        <input type="checkbox" id="includeVocals" checked>
        Include Vocals
    </label>
    
    <div id="lyricsSection" style="display: block;">
        <label>Lyrics (optional):</label>
        <textarea id="lyrics" rows="10"></textarea>
        <button onclick="generateLyrics()">Generate Lyrics</button>
    </div>
</div>

<script>
document.getElementById('includeVocals').addEventListener('change', function() {
    document.getElementById('lyricsSection').style.display = 
        this.checked ? 'block' : 'none';
});

function generateMusic() {
    const data = {
        prompt: document.getElementById('prompt').value,
        duration: parseInt(document.getElementById('duration').value),
        multistem: document.getElementById('useMultistem').checked,
        vocals: document.getElementById('includeVocals').checked,
        lyrics: document.getElementById('lyrics').value
    };
    
    fetch('/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(data => {
        playAudio(data.audio_file);
        
        if (data.stems) {
            showStems(data.stems);
        }
    });
}
</script>
```

## Workflow Improvements

### Complete Song Generation Workflow

```python
from src.services.style_analyzer import StyleAnalyzer
from src.services.lyrics_generator import LyricsGenerator
from src.services.multi_stem_generator import MultiStemGenerator

# 1. Analyze reference songs
analyzer = StyleAnalyzer()
style_profile = analyzer.analyze_subset(
    audio_files=['song1.mp3', 'song2.mp3', ...],
    profile_name='my_style'
)

# 2. Generate lyrics based on style
lyrics_gen = LyricsGenerator()
lyrics_gen.load_corpus(['transcript1.json', 'transcript2.json'])
lyrics = lyrics_gen.generate_full_song(
    style_profile=style_profile,
    structure=['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus']
)

# 3. Generate multi-stem music
generator = MultiStemGenerator(
    model_size="large",
    vocal_model="bark"
)

# Build prompt from style
descriptors = analyzer.features_to_prompt_descriptors(style_profile)
prompt = ", ".join(descriptors[:10])

# Generate
mixed_file, stems = generator.generate(
    prompt=prompt,
    duration=30,
    style_profile=style_profile,
    artist_name="Yeat",
    lyrics=lyrics,
    stems_to_generate=['vocals', 'drums', 'bass', 'melody_1', 'melody_2'],
    save_individual_stems=True,
    temperature=1.0,
    guidance_scale=3.5
)

print(f"🎵 Complete song generated!")
print(f"   Mixed: {mixed_file}")
print(f"   Stems: {list(stems.keys())}")
```

## Troubleshooting

### Bark Installation Issues

If Bark fails to install:

```bash
# Try installing dependencies manually
pip install encodec
pip install transformers
pip install scipy
pip install git+https://github.com/suno-ai/bark.git
```

### Out of Memory

If you run out of memory:

1. **Generate fewer stems at once:**
   ```python
   stems_to_generate=['drums', 'bass']  # Instead of all 5
   ```

2. **Use smaller model:**
   ```python
   generator = MultiStemGenerator(model_size="medium")
   ```

3. **Generate stems separately:**
   ```python
   stems = {}
   for stem_type in ['drums', 'bass', 'melody_1']:
       audio = generator.generate_stem(
           stem_type=stem_type,
           base_prompt=prompt,
           duration=10
       )
       stems[stem_type] = audio
       # Clear cache
       torch.cuda.empty_cache()  # If using GPU
   
   mixed = generator.mix_stems(stems)
   ```

### Bark Voice Quality

If Bark vocals sound robotic:

1. **Try different voice presets:**
   ```python
   # List available voices
   from src.services.vocal_generator import VocalGenerator
   voc_gen = VocalGenerator(model_type="bark")
   voices = voc_gen.list_bark_voices()
   
   # Test each one
   for voice in voices[:3]:
       generator.vocal_generator.voice_preset = voice
       # Generate...
   ```

2. **Adjust lyrics formatting:**
   - Add more pauses: "..."
   - Use emphasis: "WORD"
   - Add music cues: "[music]"

3. **Use MusicGen Melody fallback:**
   ```python
   generator = MultiStemGenerator(
       vocal_model="musicgen_melody"  # More musical but no lyrics
   )
   ```

## Performance Tips

1. **Parallel Stem Generation (if you have GPU):**
   ```python
   # Not yet implemented, but could use torch.multiprocessing
   ```

2. **Cache Style Profiles:**
   ```python
   # Don't re-analyze same songs
   profile = analyzer.load_named_profile("my_style")
   ```

3. **Use Shorter Test Generations:**
   ```python
   # Test with 5-8 seconds first
   mixed_file, _ = generator.generate(duration=8)
   ```

4. **Disable Stem Saving for Tests:**
   ```python
   mixed_file, _ = generator.generate(
       save_individual_stems=False  # Faster
   )
   ```

## Next Steps

After multi-stem is working:

1. **Add RVC Voice Conversion** - Clone specific artist voices
2. **Implement Advanced Mixing** - EQ, compression, reverb
3. **Add Melody Conditioning** - Guide vocals with reference melody
4. **Cascaded Generation** - Structure first, then details
5. **Real-time Preview** - Stream audio as it generates

## Summary

The multi-stem approach transforms your system from:
- ❌ Generic instrumental beats → ✅ Complete songs with vocals
- ❌ One-size-fits-all → ✅ Per-element control
- ❌ Fixed output → ✅ Remixable stems

This is a major step toward Suno-level quality! 🚀
