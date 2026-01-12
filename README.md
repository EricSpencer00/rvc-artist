# RVC Artist - Multi-Stem Music Generation

🎵 High-quality AI music generation using multi-stem synthesis. Generate trap/rap music with professional mixing and mastering using Suno v4.5+ style features.

## Quick Start

```bash
# Activate virtual environment
source .venv311/bin/activate

# Generate music
python3 app.py generate --prompt "aggressive trap beat, heavy 808s, dark atmosphere"

# Analyze audio style
python3 app.py analyze --directory data/audio

# Generate lyrics
python3 app.py lyrics --section verse

# Run tests
python3 app.py test

# Show system info
python3 app.py info
```

## Features

- **🎼 Multi-Stem Generation**: Generate vocals, drums, bass, melodies separately then mix
### Install Dependencies

```bash
# Create virtual environment (if not exists)
python3 -m venv .venv311
source .venv311/bin/activate

# Install requirements
pip install -r requirements.txt

# Install AudioCraft and PyTorch
pip install audiocraft torch torchaudio
```

## Usage

### Generate Music

```bash
source .venv311/bin/activate

# Basic generation
python3 app.py generate --prompt "aggressive trap beat, 808s"

# With all options
python3 app.py generate \
  --prompt "melodic rap, emotional vibes" \
  --duration 30 \
  --artist "Yeat" \
  --stems "drums,bass,melody_1,melody_2" \
  --no-parallel  # Disable for slower machines
```

### Analyze Audio

```bash
python3 app.py analyze --directory data/audio
```

Extracts tempo, key, energy, and creates style profile at `data/features/style_profile.json`.

### Generate Lyrics

```bash
python3 app.py lyrics --section verse --length 16 --scheme aabb
```

Sections: `intro`, `verse`, `chorus`, `bridge`, `outro`
Schemes: `aabb`, `abab`, `natural`

### Download from YouTube

```bash
python3 app.py download --url "https://youtube.com/watch?v=..."
```

### Run Tests

```bash
python3 app.py test
```

Integration tests verify all components work correctly.

### System Info

```bash
python3 app.py info
```

Shows installed dependencies and available resources.

## Architecture

```
MultiStemGenerator
├── StemProcessor          (EQ, compression, reverb per stem)
├── MasteringProcessor     (LUFS normalization, stereo width)
└── VocalGenerator         (Bark TTS synthesis)
```

### Generation Process

1. **Individual Stem Generation**: Each stem (drums, bass, melody) generated separately
2. **Per-Stem Processing**: EQ, compression, reverb applied
3. **Mixing**: Stems combined with optimized levels
4. **Mastering**: LUFS normalization, limiting, stereo widening
5. **Export**: 32kHz WAV files saved

## Configuration

### Default Settings

- Model: `large` (AudioCraft/MusicGen)
- Processing: Enabled
- Mastering: Enabled
- Parallel generation: Enabled (2 stems max)

### For Lower-RAM Machines

```bash
python3 app.py generate \
  --prompt "..." \
  --no-parallel  # Sequential generation (slower)
```

## Project Structure

```
rvc-artist/
├── app.py                    # Main CLI
├── requirements.txt          # Dependencies
├── src/
│   └── services/
│       ├── multi_stem_generator.py    # Main synthesis engine
│       ├── style_analyzer.py          # Feature extraction
│       ├── lyrics_generator.py        # Lyric generation
│       ├── vocal_generator.py         # Vocal synthesis
│       └── ...                        # Other services
├── tests/
│   └── test_battle_pipeline.py        # Integration tests
├── data/
│   ├── audio/              # Downloaded audio
│   ├── features/           # Style profiles
│   └── transcripts/        # Transcriptions
└── output/
    └── generated/          # Generated audio
```

## Performance

### Hardware Recommendations

| RAM | Performance | Config |
|-----|-------------|--------|
| 16GB | Slow | Single stem, small model |
| 32GB | Good | Single stem, large model |
| 64GB+ | Fast | Parallel (2-4 stems), large model |

### Generation Time (30s song, large model, 64GB RAM)

- Sequential (1 stem): ~120s
- Parallel (2 stems): ~65s (~2x speedup)
- Parallel (4 stems): ~45s (limits diminish)

## Troubleshooting

### ModuleNotFoundError: torch
```bash
pip install torch torchaudio
```

### "Unable to find python bindings" (DCGM)
Harmless warning on non-NVIDIA systems. Can be ignored.

### Slow generation
- Check `--no-parallel` is not set
- Verify `.venv311` has all dependencies
- Reduce duration or stem count

### Out of memory
- Use `--no-parallel` for sequential generation
- Reduce stem count
- Use smaller model size

## License

MIT

## Contact

For issues and questions, please open an issue on GitHub.


## License

This project is for educational and research purposes. Respect copyright and licensing of source materials.

## Acknowledgments

- OpenAI Whisper for transcription
- Meta AudioCraft for music generation
- Genius for lyrics API
- yt-dlp for YouTube downloading
- librosa for audio analysis

## Support

For issues and questions:
- Check existing issues on GitHub
- Review documentation above
- Create a new issue with details

---

Built with ❤️ for music AI enthusiasts
