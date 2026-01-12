# RVC Artist - Quick Usage Guide

## Setup

```bash
# Activate virtual environment
source .venv311/bin/activate

# Show available commands
python3 app.py --help
```

## Generate Music

```bash
python3 app.py generate --prompt "aggressive trap beat, heavy 808s, dark atmosphere"

# With more options
python3 app.py generate \
  --prompt "melodic rap, emotional vibes" \
  --duration 30 \
  --artist "Yeat" \
  --stems "drums,bass,melody_1,melody_2"
```

**Output:** `output/generated/mixed_YYYYMMDD_HHMMSS.wav`

## Analyze Audio

```bash
python3 app.py analyze --directory data/audio

# Default directory is data/audio
python3 app.py analyze
```

**Output:** `data/features/style_profile.json`

## Generate Lyrics

```bash
python3 app.py lyrics --section verse --length 16

# Options
python3 app.py lyrics \
  --section verse \
  --length 16 \
  --scheme aabb
```

**Sections:** intro, verse, chorus, bridge, outro
**Schemes:** aabb, abab, natural

## Download from YouTube

```bash
python3 app.py download --url "https://youtube.com/watch?v=..."

# Specify output directory
python3 app.py download \
  --url "https://youtube.com/watch?v=..." \
  --output-dir data/audio
```

**Output:** `data/audio/[filename].mp3`

## Run Tests

```bash
python3 app.py test
```

Runs all integration tests to verify pipeline integrity.

## System Info

```bash
python3 app.py info
```

Shows installed dependencies and available resources.

## Common Workflows

### Generate Music from Your Style

```bash
# 1. Analyze existing audio
python3 app.py analyze --directory data/audio

# 2. Generate new music
python3 app.py generate \
  --prompt "similar style" \
  --artist "Your Artist"
```

### Complete Song Creation

```bash
# 1. Generate base track
python3 app.py generate --prompt "trap beat" --duration 30

# 2. Generate lyrics for verse
python3 app.py lyrics --section verse

# 3. Generate lyrics for chorus
python3 app.py lyrics --section chorus
```

### Batch Download & Analyze

```bash
# Download from YouTube
python3 app.py download --url "https://youtube.com/watch?v=xxx"

# Analyze downloaded audio
python3 app.py analyze

# Generate in learned style
python3 app.py generate --prompt "in the style of artist"
```

## Performance Tips

### For 64GB RAM (Recommended)
```bash
# Parallel generation enabled by default
python3 app.py generate --prompt "..."
```

### For Slower Machines
```bash
# Disable parallel generation
python3 app.py generate \
  --prompt "..." \
  --no-parallel
```

### Adjust Quality vs Speed
```bash
# Faster generation (lower quality)
python3 app.py generate \
  --prompt "..." \
  --temperature 0.8 \
  --guidance 2.5

# Higher quality (slower)
python3 app.py generate \
  --prompt "..." \
  --temperature 1.2 \
  --guidance 4.0
```

## Parameter Guide

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--duration` | 30 | 5-120 | Seconds |
| `--temperature` | 1.0 | 0.5-1.5 | Lower = more consistent, Higher = more creative |
| `--guidance` | 3.5 | 2.0-5.0 | Lower = faster, Higher = more prompt adherence |
| `--stems` | drums,bass,melody_1 | - | Comma-separated stem types |

## Troubleshooting

**"ModuleNotFoundError: audiocraft"**
```bash
pip install audiocraft
```

**Slow generation**
- Check you're using `.venv311`
- Verify `--no-parallel` is not set
- Reduce duration

**Memory errors**
- Use `--no-parallel`
- Reduce stem count
- Reduce duration

**File not found**
- Check directory paths exist
- Use absolute paths if needed
- Verify file extensions (.mp3, .wav, etc.)

## Output Files

### Generated Music
```
output/generated/
├── mixed_YYYYMMDD_HHMMSS.wav      # Final mixed track
├── stem_drums_YYYYMMDD_HHMMSS.wav # Individual drum stem
├── stem_bass_YYYYMMDD_HHMMSS.wav  # Individual bass stem
└── stem_melody_1_YYYYMMDD_HHMMSS.wav # Individual melody
```

### Style Profiles
```
data/features/
└── style_profile.json             # Extracted style features
```

### Downloaded Audio
```
data/audio/
└── [filename].mp3                 # Downloaded from YouTube
```

## Next Steps

1. Run `python3 app.py info` to verify setup
2. Try `python3 app.py test` to run integration tests
3. Generate your first track with `python3 app.py generate`
4. Analyze your music with `python3 app.py analyze`
5. Generate lyrics with `python3 app.py lyrics`

For more details, see [README.md](README.md)
