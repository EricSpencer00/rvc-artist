"""
Multi-Stem Music Generator
==========================
Generates music by creating separate stems (vocals, drums, bass, melody)
then mixing them together for higher quality output.

This approach is inspired by Suno's architecture and solves the "generic beat" problem
by giving each element dedicated generation attention.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import soundfile as sf

try:
    from .vocal_generator import VocalGenerator
    VOCAL_GENERATOR_AVAILABLE = True
except ImportError:
    VOCAL_GENERATOR_AVAILABLE = False
    print("VocalGenerator not available - vocals will be instrumental only")


class MultiStemGenerator:
    """Generates music using separate stem generation and mixing."""
    
    STEM_TYPES = {
        'vocals': {
            'description': 'singing voice, vocal melody, human voice',
            'instruments': None,  # Will use dedicated vocal model
            'eq_boost': [1000, 3000],  # Boost presence frequencies
            'priority': 1,  # Generate first
            'use_vocal_generator': True  # Use VocalGenerator if available
        },
        'drums': {
            'description': 'drum kit, percussion, hi-hats, snare, kick',
            'instruments': ['drums', 'percussion'],
            'eq_boost': [60, 100, 8000],  # Boost sub and high frequencies
            'priority': 2
        },
        'bass': {
            'description': 'bass, sub bass, 808',
            'instruments': ['bass'],
            'eq_boost': [60, 120],  # Boost sub frequencies
            'priority': 3
        },
        'melody_1': {
            'description': 'lead synth, bells, melodic elements',
            'instruments': ['synth', 'bells', 'lead'],
            'eq_boost': [2000, 4000],
            'priority': 4
        },
        'melody_2': {
            'description': 'pads, atmospheric synth, background melody',
            'instruments': ['synth', 'pad', 'strings'],
            'eq_boost': [500, 2000],
            'priority': 5
        }
    }
    
        self,
        model_size: str = "large",
        use_stereo: bool = True,
        vocal_model: str = "bark"
    ):
        """
        Initialize multi-stem generator.
        
        Args:
            model_size: MusicGen model size
            use_stereo: Whether to use stereo models (if available)
            vocal_model: Vocal synthesis model ('bark', 'musicgen_melody', None)
        """
        self.model_size = model_size
        self.use_stereo = use_stereo
        self.vocal_model_type = vocal_model
        self.models = {}
        self.vocal_generator = None
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self._load_models()
        
        # Load vocal generator if available and requested
        if vocal_model and VOCAL_GENERATOR_AVAILABLE:
            try:
                self.vocal_generator = VocalGenerator(
                    model_type=vocal_model,
                    device=self.device
                )
            except Exception as e:
                print(f"Could not load vocal generator: {e}"
        self._load_models()
    
    def _load_models(self):
        """Load MusicGen models for different stems."""
        try:
            from audiocraft.models import MusicGen
            
            # Load main instrumental model
            print(f"Loading MusicGen for instrumental stems...")
            if self.device == "mps":
                self.models['instrumental'] = MusicGen.get_pretrained(
                    self.model_size, device="cpu"
                )
                self.device = "cpu"
            else:
                self.models['instrumental'] = MusicGen.get_pretrained(
                    self.model_size, device=self.device
                )
            
            # Try to load melody model for vocals (better for melodic content)
            try:
                print(f"Loading MusicGen Melody for vocals...")
                if self.device == "mps" or self.device == "cpu":
                    self.models['vocals'] = MusicGen.get_pretrained(
                        'melody', device="cpu"
                    )
                else:
                    self.models['vocals'] = MusicGen.get_pretrained(
                        'melody', device=self.device
                    )
            except:
                print("Melody model not available, using standard model for vocals")
                self.models['vocals'] = self.models['instrumental']
            
            print(f"✅ Multi-stem models loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = {}
    
    def _build_stem_prompt(
        self,
        base_prompt: str,
        stem_type: str,
        style_profile: Optional[Dict] = None
    ) -> str:
        """
        Build a specialized prompt for a specific stem.
        
        Args:
            base_prompt: Base style description
            stem_type: Type of stem (vocals, drums, bass, melody_1, melody_2)
            style_profile: Optional style profile for additional context
            
        Returns:
            Stem-specific prompt
        """
        stem_info = self.STEM_TYPES[stem_type]
        
        # Extract style elements from base prompt
        parts = [base_prompt]
        
        # Add stem-specific description
        parts.append(stem_info['description'])
        
        # Add focus instruction
        parts.append(f"focus on {stem_type} only")
        
        # Add style-specific modifiers
        if style_profile:
            if stem_type == 'drums':
                energy = style_profile.get('energy', {}).get('mean', 0.5)
                if energy > 0.7:
                    parts.append("aggressive hard-hitting drums")
                else:
                    parts.append("smooth groovy drums")
                    
            elif stem_type == 'bass':
                spectral = style_profile.get('spectral', {})
        lyrics: Optional[str] = None,
        artist_name: Optional[str] = None,
                brightness = spectral.get('spectral_centroid_mean', 2000)
                if brightness < 1500:
                    parts.append("deep sub bass")
                else:
                    parts.append("mid bass with harmonics")
                    
            elif stem_type in ['melody_1', 'melody_2']:
                key = style_profile.get('key', {}).get('most_common', 'C major')
                parts.append(f"in {key}")
        
        return ", ".join(parts)
    
    def generate_stem(
        self,
        stem_type: str,
        base_prompt: str,
        duration: int,
        style_profile: Optional[Dict] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_scale: float = 3.0
    ) -> np.ndarray:
        """
        Generate a single stem.
        
        Args:
            stem_type: Type of stem to generate
            base_prompt: Base style prompt
            lyrics: Optional lyrics for vocal stems
            artist_name: Artist name for style
            temperature: Sampling temperature
            top_k: Top-k sampling
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Audio array for the stem
        """
        # Special handling for vocals with VocalGenerator
        if stem_type == 'vocals' and self.vocal_generator and lyrics:
            print(f"\n🎤 Generating vocals with {self.vocal_model_type}...")
            
            try:
                # Determine vocal style from base prompt
                if 'aggressive' in base_prompt.lower() or 'rage' in base_prompt.lower():
                    vocal_style = 'aggressive'
                elif 'melodic' in base_prompt.lower():
                    vocal_style = 'melodic_rap'
                else:
                    vocal_style = 'rap'
                
                # Generate vocals using VocalGenerator
                audio_file = self.vocal_generator.generate(
                    lyrics=lyrics,
                    style=vocal_style,
                    duration=duration,
                    style_profile=style_profile,
                    artist_name=artist_name,
                    save_audio=False  # We'll handle saving in main generate()
                )
                
                # Load the audio if it was saved
                if audio_file and Path(audio_file).exists():
                    import soundfile as sf
                    audio, sr = sf.read(audio_file)
                    
                    # Resample to 32kHz if needed
                    if sr != 32000:
                        from scipy import signal
                        num_samples = int(len(audio) * 32000 / sr)
                        audio = signal.resample(audio, num_samples)
                    
                    return audio
                else:
                    # Audio returned directly (not saved)
                    # Bark returns at 24kHz, need to resample
                    print("Using direct audio output from vocal generator")
                    # Will fall through to instrumental vocal generation
                    
            except Exception as e:
                print(f"Vocal generation failed, falling back to instrumental: {e}")
        
        # Choose the right model for instrumental stemsstem
        """
        # Choose the right model
        if stem_type == 'vocals' and 'vocals' in self.models:
            model = self.models['vocals']
        else:
            model = self.models.get('instrumental', self.models.get('vocals'))
        
        if model is None:
            print(f"No model available for {stem_type}, generating silence")
            sample_rate = 32000  # MusicGen default
            return np.zeros(int(duration * sample_rate))
        
        # Build stem-specific prompt
        prompt = self._build_stem_prompt(base_prompt, stem_type, style_profile)
        print(f"\n🎵 Generating {stem_type}: '{prompt}'")
        
        try:
            # Set generation parameters
            model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                cfg_coef=guidance_scale  # Classifier-free guidance
            )
            
            # Generate
            descriptions = [prompt]
            wav = model.generate(descriptions, progress=False)
            
            # Convert to numpy
            audio = wav.squeeze(0).cpu().numpy()
            
            # If stereo, keep both channels; if mono, return as is
            if audio.ndim == 2:
                # Stereo: (2, samples)
                return audio
            else:
                # Mono: (samples,)
                return audio
                
        except Exception as e:
            print(f"Error generating {stem_type}: {e}")
            sample_rate = 32000
            return np.zeros(int(duration * sample_rate))
    
    def _apply_stem_processing(
        self,
        audio: np.ndarray,
        stem_type: str,
        sample_rate: int = 32000
    ) -> np.ndarray:
        """
        Apply stem-specific processing (EQ, compression, etc.).
        
        Args:
            audio: Input audio
            stem_type: Type of stem
            sample_rate: Sample rate
            
        Returns:
            Processed audio
        """
        # For now, just normalize
        # In future: Add EQ, compression, stereo widening
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9  # Normalize to -1dB
        
        return audio
    
    def mix_stems(
        self,
        stems: Dict[str, np.ndarray],
        mix_levels: Optional[Dict[str, float]] = None,
        sample_rate: int = 32000
    ) -> np.ndarray:
        """
        Mix multiple stems together with level balancing.
        
        Args:
            stems: Dictionary of stem_type -> audio_array
            mix_levels: Optional level multipliers for each stem
            sample_rate: Sample rate
            
        Returns:
            Mixed audio
        """
        if not stems:
            return np.zeros(int(sample_rate * 5))
        
        # Default mix levels (optimized for typical music)
        default_levels = {
            'vocals': 1.0,      # Vocals loudest
            'drums': 0.85,      # Drums slightly lower
            'bass': 0.9,        # Bass prominent
            'melody_1': 0.7,    # Lead melody
            'melody_2': 0.5     # Background elements quieter
        }
        
        if mix_levels is None:
            mix_levels = default_levels
        
        # Find max length
        max_length = max(
            audio.shape[-1] if audio.ndim > 1 else len(audio)
            for audio in stems.values()
        )
        
        # Determine if we're mixing stereo or mono
        is_stereo = any(audio.ndim == 2 for audio in stems.values())
        
        if is_stereo:
            mixed = np.zeros((2, max_length))
        else:
            mixed = np.zeros(max_length)
        lyrics: Optional[str] = None,
        
        # Mix each stem
        for stem_type, audio in stems.items():
            level = mix_levels.get(stem_type, 1.0)
            
            # Handle mono/stereo mismatch
            if is_stereo and audio.ndim == 1:
                # Convert mono to stereo
                audio = np.stack([audio, audio])
            elif not is_stereo and audio.ndim == 2:
                # Convert stereo to mono
                audio = audio.mean(axis=0)
            
            # Pad if needed
            lyrics: Optional lyrics for vocal generation
            current_length = audio.shape[-1] if audio.ndim > 1 else len(audio)
            if current_length < max_length:
                if audio.ndim == 2:
                    padding = np.zeros((2, max_length - current_length))
                    audio = np.concatenate([audio, padding], axis=1)
                else:
                    padding = np.zeros(max_length - current_length)
                    audio = np.concatenate([audio, padding])
            
            # Mix with level
            if audio.ndim == 2:
                mixed += audio * level
            else:
                mixed += audio * level
        
        # Master bus processing: soft limiting
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            # Soft clipping
            mixed = np.tanh(mixed / max_val) * 0.95
        else:
            # Normalize to use more headroom
            mixed = mixed * 0.95
        
        return mixed
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        style_profile: Optional[Dict] = None,
        artist_name: Optional[str] = None,
        output_dir: str = "./output/generated",
        stems_to_generate: Optional[List[str]] = None,
        save_individual_stems: bool = True,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_scale: float = 3.0
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate complete song with multiple stems.
        
        Args:
            prompt: Base style prompt
            duration: Duration in seconds
            style_profile: Style profile for guidance
            artist_name: Artist name for prompt enhancement
            output_dir: Output directory
            stems_to_generate: Which stems to generate (None = all)
            save_individual_stems: Whether to save individual stem files
            temperature: Sampling temperature
            top_k: Top-k sampling
            guidance_scale: Guidance scale for generation
            
        Returns:
            (mixed_file_path, {stem_type: stem_file_path})
        """
        if stems_to_generate is None:
            stems_to_generate = list(self.STEM_TYPES.keys())
        
        # Build base prompt
        if artist_name:
            prompt = f"{prompt}, in the style of {artist_name}"
        
        print(f"\n{'='*60}")
        print(f"🎼 MULTI-STEM GENERATION")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Duration: {duration}s")
        print(f"Stems: {', '.join(stems_to_generate)}")
        print(f"{'='*60}\n")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate each stem
        stems = {}
        stem_paths = {}
        
        # Sort stems by priority
        sorted_stems = sorted(
            stems_to_generate,
            key=lambda x: self.STEM_TYPES[x]['priority']
        )
        
        for stem_type in sorted_stems:
            audio = self.generate_stem(
                stem_type=stem_type,
                base_prompt=prompt,
                lyrics=lyrics if stem_type == 'vocals' else None,
                artist_name=artist_name,
                duration=duration,
                style_profile=style_profile,
                temperature=temperature,
                top_k=top_k,
                guidance_scale=guidance_scale
            )
            
            # Apply stem-specific processing
            audio = self._apply_stem_processing(
                audio, stem_type, sample_rate=32000
            )
            
            stems[stem_type] = audio
            
            # Save individual stem
            if save_individual_stems:
                stem_file = output_path / f"stem_{stem_type}_{timestamp}.wav"
                
                # Ensure audio is in right shape for soundfile
                if audio.ndim == 2:
                    # Stereo: transpose to (samples, channels)
                    audio_to_save = audio.T
                else:
                    # Mono: keep as is
                    audio_to_save = audio
                
                sf.write(stem_file, audio_to_save, 32000)
                stem_paths[stem_type] = str(stem_file)
                print(f"✅ Saved {stem_type} stem")
        
        # Mix stems
        print(f"\n🎚️  Mixing {len(stems)} stems...")
        mixed_audio = self.mix_stems(stems, sample_rate=32000)
        
        # Save mixed output
        mixed_file = output_path / f"mixed_{timestamp}.wav"
        
        if mixed_audio.ndim == 2:
            mixed_to_save = mixed_audio.T
        else:
            mixed_to_save = mixed_audio
            
        sf.write(mixed_file, mixed_to_save, 32000)
        
        # Save metadata
        metadata = {
            'prompt': prompt,
            'duration': duration,
            'model_size': self.model_size,
            'stems_generated': list(stems.keys()),
            'temperature': temperature,
            'top_k': top_k,
            'guidance_scale': guidance_scale,
            'generated_at': timestamp,
            'style_profile_used': bool(style_profile),
            'artist_name': artist_name
        }
        
        metadata_file = mixed_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Mixed output: {mixed_file}")
        print(f"Individual stems: {len(stem_paths)}")
        print(f"{'='*60}\n")
        
        return str(mixed_file), stem_paths


if __name__ == "__main__":
    # Test the multi-stem generator
    generator = MultiStemGenerator(model_size="large")
    
    test_prompt = "aggressive trap beat, heavy 808s, dark atmosphere"
    
    mixed_path, stem_paths = generator.generate(
        prompt=test_prompt,
        duration=10,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1'],
        save_individual_stems=True
    )
    
    print(f"Test complete! Mixed file: {mixed_path}")
