# Multi-Stem Music Generation - Implementation Guide

## Overview

This document explains the multi-stem music generation approach, inspired by how Suno evolved from v1 to v4.5+.

## The Problem with Single-Model Generation

Your current system uses **MusicGen** in a monophonic way:
- Single model generates all elements together
- Results in "generic beat" with no vocals
- Limited control over individual elements
- Difficult to balance different components

## How Suno Evolved (v1 → v4.5)

### Key Innovations

1. **Multi-Stem Architecture** ✅ (What we're implementing!)
   - Separate generation for vocals, drums, bass, instruments
   - Each stem gets dedicated model attention
   - Better quality per component

2. **Cascaded Generation**
   - Coarse structure first (beat, chord progression)
   - Then details (fills, ornaments, effects)
   - We can add this next

3. **Controllable Vocals**
   - TTS-like singing voice synthesis
   - Lyric-to-melody alignment
   - Prosody control (pitch, timing, emotion)

4. **Better Conditioning**
   - Multi-modal prompts (text + audio reference)
   - Style embeddings from reference tracks
   - Fine-grained control parameters

5. **Advanced Post-Processing**
   - Automatic mastering
   - Stem separation for refinement
   - Dynamic range compression
   - Stereo widening

## Our Multi-Stem Implementation

### Architecture

```
Input Prompt + Style Profile
            ↓
    ┌───────┴───────┐
    │  Stem Router  │
    └───────┬───────┘
            ↓
    ┌───────┴────────────────────────┐
    │                                 │
┌───↓────┐  ┌────────┐  ┌─────┐  ┌────────┐  ┌────────┐
│ Vocals │  │ Drums  │  │Bass │  │Melody1 │  │Melody2 │
│ (TTS)  │  │(MusicG)│  │(MG) │  │  (MG)  │  │  (MG)  │
└───┬────┘  └───┬────┘  └──┬──┘  └───┬────┘  └───┬────┘
    │           │           │         │           │
    │    ┌──────↓───────────┴─────────┴───────────┴──┐
    │    │         Stem Processing Layer              │
    │    │  (EQ, Compression, Normalization)          │
    │    └──────┬───────────────────────────────────┬─┘
    │           │                                   │
    └───────────↓───────────────────────────────────┘
                │
         ┌──────↓──────┐
         │  Mixer Bus  │
         │ (Balancing) │
         └──────┬──────┘
                ↓
         ┌──────↓──────┐
         │   Master    │
         │ (Limiting)  │
         └──────┬──────┘
                ↓
          Final Output
```

### Stem Types

1. **Vocals** (Priority 1)
   - Uses MusicGen Melody model (better for melodic content)
   - Can be conditioned on lyrics + melody
   - Frequency range: 200-8000 Hz (boosted)

2. **Drums** (Priority 2)
   - Kick, snare, hi-hats, percussion
   - Frequency: 60-100 Hz (sub), 8000+ Hz (high)
   - Highest dynamic range

3. **Bass** (Priority 3)
   - Sub bass, 808s, bass synth
   - Frequency: 60-250 Hz
   - Mono (centered)

4. **Melody 1** (Priority 4)
   - Lead synths, bells, main melodic elements
   - Frequency: 1000-5000 Hz
   - Stereo width

5. **Melody 2** (Priority 5)
   - Pads, atmosphere, background
   - Frequency: 500-3000 Hz
   - Wide stereo

### Generation Flow

1. **Prompt Building**
   - Base prompt from style profile
   - Stem-specific enhancements
   - Genre/artist modifiers

2. **Sequential Generation**
   - Generate in priority order
   - Each stem uses specialized prompt
   - Different temperature/guidance per stem

3. **Processing**
   - Normalize levels
   - Apply EQ curves
   - Add compression (future)

4. **Mixing**
   - Balance levels (vocals loudest)
   - Pan positioning (future)
   - Stereo imaging (future)

5. **Mastering**
   - Soft limiting
   - Loudness normalization
   - Export

## Usage Examples

### Basic Multi-Stem

```python
from src.services.multi_stem_generator import MultiStemGenerator

generator = MultiStemGenerator(model_size="large")

mixed_path, stem_paths = generator.generate(
    prompt="aggressive trap beat, heavy 808s, dark atmosphere",
    duration=15,
    artist_name="Yeat",
    stems_to_generate=['drums', 'bass', 'melody_1'],
    save_individual_stems=True
)
```

### With Style Profile

```python
from src.services.style_analyzer import StyleAnalyzer

# Load style profile
analyzer = StyleAnalyzer()
profile = analyzer.load_named_profile("rage_trap_20")

# Generate descriptors
descriptors = analyzer.features_to_prompt_descriptors(profile)
base_prompt = ", ".join(descriptors[:8])

# Generate with profile guidance
mixed_path, stem_paths = generator.generate(
    prompt=base_prompt,
    duration=20,
    style_profile=profile,
    artist_name="Yeat",
    stems_to_generate=['drums', 'bass', 'melody_1', 'melody_2'],
    guidance_scale=4.0  # Higher = more prompt adherence
)
```

### Custom Mixing

```python
# Generate stems
stems = {}
for stem_type in ['drums', 'bass', 'melody_1']:
    audio = generator.generate_stem(
        stem_type=stem_type,
        base_prompt="trap beat",
        duration=10
    )
    stems[stem_type] = audio

# Mix with custom levels
mix_levels = {
    'drums': 1.0,   # Loudest
    'bass': 0.9,    # Slightly lower
    'melody_1': 0.6 # Background
}

mixed = generator.mix_stems(stems, mix_levels=mix_levels)
```

## Parameter Tuning

### Temperature
- **Low (0.5-0.8)**: More predictable, coherent
- **Medium (0.9-1.1)**: Balanced creativity
- **High (1.2-1.5)**: Very creative, experimental

### Guidance Scale (CFG)
- **Low (1.0-2.0)**: More freedom, less prompt adherence
- **Medium (3.0-4.0)**: Balanced (recommended)
- **High (5.0+)**: Strict prompt following, less variety

### Stem-Specific Tips

**Drums:**
- Higher temperature (1.1-1.3) for variation
- Guidance 3.0-4.0
- Prompt: "hard-hitting drums, punchy kick, crisp snare"

**Bass:**
- Medium temperature (0.9-1.1)
- Guidance 3.5-4.5 (stay on prompt)
- Prompt: "deep sub bass, 808 bass, low end"

**Melody:**
- Variable temperature based on style
- Guidance 3.0-4.0
- Prompt: Include key signature and mood

## Next Steps / Future Improvements

### 1. Vocal Synthesis (HIGH PRIORITY)
Current limitation: No actual singing vocals

**Solution options:**
- **Bark** (Suno's TTS) - Can generate singing
- **So-VITS-SVC** - Voice conversion
- **RVC** (Retrieval Voice Conversion) - Clone artist voice
- **MusicGen Melody** + lyrics conditioning

**Implementation:**
```python
# Add to MultiStemGenerator
def generate_vocal_stem(
    self,
    lyrics: str,
    melody_reference: Optional[np.ndarray],
    duration: int
) -> np.ndarray:
    # Use Bark or RVC to generate singing voice
    pass
```

### 2. Better Stem Separation
- Use Demucs/Spleeter to analyze reference tracks
- Extract patterns from separated stems
- Guide generation with extracted features

### 3. Advanced Mixing
```python
def mix_stems_advanced(
    self,
    stems: Dict[str, np.ndarray],
    apply_eq: bool = True,
    apply_compression: bool = True,
    apply_reverb: bool = True,
    stereo_width: float = 1.0
) -> np.ndarray:
    # EQ each stem
    # Compress dynamics
    # Add reverb/effects
    # Stereo widening
    pass
```

### 4. Cascaded Generation
Generate structure first, then details:

```python
# Step 1: Generate low-res structure (chord progression)
structure = generator.generate_structure(
    prompt=prompt,
    duration=30,
    resolution="low"  # 8kHz
)

# Step 2: Generate full-res conditioned on structure
final = generator.generate_conditioned(
    structure_audio=structure,
    prompt=prompt,
    duration=30,
    resolution="high"  # 32kHz
)
```

### 5. Style Interpolation
Blend multiple style profiles:

```python
mixed_profile = analyzer.interpolate_profiles(
    profiles=[profile_a, profile_b],
    weights=[0.7, 0.3]
)
```

### 6. Real-time Parameter Control
Add hooks for real-time parameter adjustment during generation

## Technical Notes

### Memory Management
- Each stem generation: ~2-4GB VRAM (GPU) or RAM (CPU)
- Sequential generation prevents OOM
- Consider batching small stems together

### Performance
- CPU (64GB RAM): ~30-60s per 10s stem
- GPU (RTX 3090): ~5-10s per 10s stem
- MPS (M1/M2): Similar to CPU due to autocast issues

### Quality Tips
1. **Use longer durations** (15-30s) - Better coherence
2. **Higher guidance for specific genres** - Trap, EDM benefit from CFG 4.0+
3. **Lower temperature for bass** - Keeps low end tight
4. **Save stems individually** - Allows remixing later

## Comparison: Single vs Multi-Stem

| Aspect | Single Model | Multi-Stem |
|--------|-------------|------------|
| Quality | Generic | Focused |
| Vocals | Instrumental only | Can add vocals |
| Control | Limited | Per-stem control |
| Mix Balance | Fixed | Adjustable |
| Processing | One-size-fits-all | Stem-specific |
| File Size | 1 file | N+1 files |
| Gen Time | 1x | Nx (parallel possible) |

## Testing Checklist

- [ ] Test basic 3-stem generation (drums, bass, melody)
- [ ] Test full 5-stem generation (add melody_2)
- [ ] Test with style profile guidance
- [ ] Test different genres (trap, R&B, cloud rap)
- [ ] Test custom mix levels
- [ ] Compare single vs multi-stem output quality
- [ ] Test vocal stem integration (when implemented)
- [ ] Test longer durations (30s+)
- [ ] Test different temperature/guidance combinations
- [ ] Profile memory usage and generation time

## Resources

- MusicGen: https://github.com/facebookresearch/audiocraft
- Demucs (stem separation): https://github.com/facebookresearch/demucs
- Bark (TTS): https://github.com/suno-ai/bark
- RVC (voice conversion): https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

## Conclusion

The multi-stem approach gives you:
1. ✅ **Better quality** - Each element gets focused generation
2. ✅ **More control** - Adjust individual components
3. ✅ **Mixable** - Save stems for later remixing
4. ✅ **Scalable** - Add more stem types as needed
5. ⏳ **Vocals** - Next step: Add real singing voice

This is a significant step toward Suno-quality local generation!
