"""
Music Generator Service
=======================
Generates music using Meta's AudioCraft (MusicGen).
Can be guided by style profiles and text prompts.
Supports section-aware generation and blueprint-based composition.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class MusicGenerator:
    """Generates music using AudioCraft's MusicGen with enhanced style guidance."""
    
    # Section templates for structured generation
    SECTION_TEMPLATES = {
        'intro': {
            'duration': 8,
            'energy_modifier': 0.6,
            'descriptors': ['atmospheric intro', 'building anticipation', 'sparse arrangement']
        },
        'verse': {
            'duration': 20,
            'energy_modifier': 0.8,
            'descriptors': ['rhythmic verse section', 'steady groove', 'melodic development']
        },
        'pre_chorus': {
            'duration': 8,
            'energy_modifier': 0.9,
            'descriptors': ['building tension', 'rising energy', 'pre-drop build']
        },
        'chorus': {
            'duration': 16,
            'energy_modifier': 1.0,
            'descriptors': ['powerful chorus', 'full energy', 'catchy hook', 'main theme']
        },
        'hook': {
            'duration': 8,
            'energy_modifier': 1.0,
            'descriptors': ['memorable hook', 'earworm melody', 'signature phrase']
        },
        'bridge': {
            'duration': 12,
            'energy_modifier': 0.7,
            'descriptors': ['contrasting bridge', 'different texture', 'breakdown section']
        },
        'drop': {
            'duration': 16,
            'energy_modifier': 1.0,
            'descriptors': ['heavy drop', 'maximum energy', 'bass-heavy impact', 'climax']
        },
        'outro': {
            'duration': 10,
            'energy_modifier': 0.5,
            'descriptors': ['fading outro', 'winding down', 'conclusion']
        }
    }
    
    def __init__(self, model_size: str = "large"):
        """
        Initialize the generator.
        
        Args:
            model_size: MusicGen model size ('small', 'medium', 'large', 'melody')
        """
        self.model_size = model_size
        self.model = None
        
        # Determine best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self._load_model()

    def _load_model(self):
        """Load the MusicGen model."""
        try:
            from audiocraft.models import MusicGen
            print(f"Loading MusicGen model: {self.model_size} on {self.device}")
            
            # MPS has autocast issues with AudioCraft, load on CPU instead
            # With 64GB RAM, CPU inference is viable for generation
            if self.device == "mps":
                print(f"Note: MusicGen doesn't fully support MPS autocast, using CPU")
                print(f"With 64GB RAM, CPU generation is still fast enough")
                self.model = MusicGen.get_pretrained(self.model_size, device="cpu")
                self.device = "cpu"  # Update device to CPU
            else:
                self.model = MusicGen.get_pretrained(self.model_size, device=self.device)
                
            print(f"✅ MusicGen {self.model_size} model loaded successfully on {self.device}")
        except ImportError:
            print("AudioCraft not available. Install with: pip install audiocraft")
            self.model = None
        except Exception as e:
            print(f"Error loading MusicGen model: {e}")
            self.model = None
    
    def style_to_prompt(self, style_profile: Optional[Dict[str, Any]] = None, artist_name: Optional[str] = None) -> str:
        """
        Convert a style profile to a text prompt for MusicGen.
        Uses rich feature-to-text mapping for better results.
        
        Args:
            style_profile: Style profile from StyleAnalyzer
            artist_name: Optional name of the artist to mimic
            
        Returns:
            Text prompt describing the musical style
        """
        if not style_profile or not style_profile.get('characteristics'):
            if artist_name:
                return f"{artist_name} style music"
            return "pop music"
        
        chars = style_profile.get('characteristics', {})
        tempo = style_profile.get('tempo', {})
        key_info = style_profile.get('key', {})
        lyrics_info = style_profile.get('lyrics', {})
        energy = style_profile.get('energy', {})
        spectral = style_profile.get('spectral', {})
        
        # Build descriptive prompt
        prompt_parts = []
        
        # Add artist context if known
        if artist_name:
            prompt_parts.append(f"in the style of {artist_name}")
            
            # Special artist-specific enhancers
            artist_lower = artist_name.lower()
            if 'yeat' in artist_lower:
                prompt_parts.extend(["aggressive trap beat", "heavy distorted 808", "synthetic bells", "rage trap", "saturated texture"])
            elif 'playboi carti' in artist_lower or 'carti' in artist_lower:
                prompt_parts.extend(["minimalist trap", "high pitched synths", "vampire aesthetic", "f1lthy style", "dark moody pads"])
            elif 'travis scott' in artist_lower:
                prompt_parts.extend(["psychedelic trap", "dark atmospheric", "heavy reverb", "distorted vocals", "cinematic low end"])
            elif 'drake' in artist_lower:
                prompt_parts.extend(["melodic rap", "atmospheric pads", "emotional chords", "OVO sound", "smooth transitions"])
            elif 'future' in artist_lower:
                prompt_parts.extend(["dark trap", "heavy auto-tune influence", "808 patterns", "melancholic melodies"])
            elif 'metro boomin' in artist_lower or 'metro' in artist_lower:
                prompt_parts.extend(["cinematic trap", "orchestral elements", "hard hitting drums", "dark atmosphere"])
        
        # Add high-quality production descriptors (Suno style)
        prompt_parts.extend(["studio quality", "professional mix", "crisp percussion", "high fidelity"])
        
        # Add tempo-based descriptors (more specific)
        tempo_mean = tempo.get('mean', 120)
        tempo_std = tempo.get('std', 10)
        
        if tempo_mean < 80:
            prompt_parts.append("slow groove")
        elif tempo_mean < 100:
            prompt_parts.append("mid-tempo chill vibe")
        elif tempo_mean < 120:
            prompt_parts.append("steady driving rhythm")
        elif tempo_mean < 140:
            prompt_parts.append("upbeat energetic pace")
        elif tempo_mean < 160:
            prompt_parts.append("fast aggressive tempo")
        else:
            prompt_parts.append("hyper-speed intensity")
        
        prompt_parts.append(f"{int(tempo_mean)} BPM")
        
        # Add energy-based descriptors
        rms_mean = energy.get('rms_mean', 0.1)
        dynamic_range = energy.get('dynamic_range_mean', 0.1)
        
        if rms_mean > 0.15:
            prompt_parts.append("loud punchy mix")
        elif rms_mean < 0.05:
            prompt_parts.append("soft intimate production")
        
        if dynamic_range > 0.15:
            prompt_parts.append("big dynamic drops")
        
        # Add spectral/timbre descriptors
        brightness = spectral.get('brightness_mean', 2000)
        
        if brightness < 1500:
            prompt_parts.append("warm dark low-end heavy")
        elif brightness > 3000:
            prompt_parts.append("bright crisp treble")
        
        # Add key if available
        most_common_key = key_info.get('most_common', '')
        if most_common_key:
            prompt_parts.append(f"in {most_common_key}")
        
        # Add overall characteristics
        if chars.get('overall'):
            prompt_parts.append(chars['overall'])

        # Add lyrical themes / keywords (top 5)
        keywords = lyrics_info.get('top_keywords', [])
        if keywords:
            # Map keywords to musical descriptors
            keyword_mood_map = {
                'money': 'luxury flex vibes',
                'love': 'emotional melodic feel',
                'dark': 'dark moody atmosphere',
                'fast': 'high energy intensity',
                'night': 'nocturnal mysterious vibe',
                'gang': 'street aggressive energy',
                'pain': 'melancholic emotional depth',
            }
            
            for kw in keywords[:3]:
                if kw.lower() in keyword_mood_map:
                    prompt_parts.append(keyword_mood_map[kw.lower()])
            
            # Also include raw themes
            theme_str = ", ".join(keywords[:4])
            prompt_parts.append(f"themes of {theme_str}")
        
        prompt = ", ".join(prompt_parts) if prompt_parts else "pop music"
        
        return prompt

    def build_section_prompt(
        self,
        base_prompt: str,
        section_type: str,
        energy_level: float = 1.0
    ) -> str:
        """
        Build a prompt for a specific song section.
        
        Args:
            base_prompt: Base style prompt
            section_type: Type of section (intro, verse, chorus, etc.)
            energy_level: Energy modifier (0.0 to 1.0)
            
        Returns:
            Section-specific prompt
        """
        section_info = self.SECTION_TEMPLATES.get(section_type, self.SECTION_TEMPLATES['verse'])
        descriptors = section_info['descriptors']
        
        # Modify energy descriptor
        if energy_level < 0.5:
            energy_desc = "low energy, calm"
        elif energy_level < 0.8:
            energy_desc = "moderate energy, building"
        else:
            energy_desc = "high energy, intense"
        
        section_prompt = f"{base_prompt}, {', '.join(descriptors)}, {energy_desc}"
        
        return section_prompt
    
    def generate(
        self,
        prompt: Optional[str] = None,
        duration: int = 30,
        style_profile: Optional[Dict[str, Any]] = None,
        artist_name: Optional[str] = None,
        output_dir: str = "./output/generated",
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0
    ) -> str:
        """
        Generate music based on a prompt and/or style profile.
        
        Args:
            prompt: Text description of desired music (optional)
            duration: Duration in seconds
            style_profile: Style profile to guide generation (optional)
            artist_name: Artist name to mimic (optional)
            output_dir: Directory to save generated audio
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Path to generated audio file
        """
        # Build the prompt first
        if not prompt:
            if style_profile:
                prompt = self.style_to_prompt(style_profile, artist_name=artist_name)
            else:
                prompt = f"{artist_name} style" if artist_name else "pop music"
        
        print(f"--- ENHANCED PROMPT: {prompt} ---")

        if self.model is None:
            return self._fallback_generate(prompt, duration, output_dir)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating music with prompt: '{prompt}'")
        print(f"Duration: {duration} seconds")
        print(f"Model: {self.model_size}, Device: {self.device}")
        
        try:
            # Set generation parameters
            self.model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Generate
            descriptions = [prompt]
            wav = self.model.generate(descriptions, progress=True)
            
            # Save the audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.wav"
            output_file = output_path / filename

            # Convert tensor to CPU for serialization but keep it as torch.Tensor
            from audiocraft.data.audio import audio_write
            audio_tensor = wav.squeeze(0).cpu()

            # Save using audiocraft's audio_write (handles normalization)
            audio_write(
                str(output_file.with_suffix('')),  # without extension
                audio_tensor,
                self.model.sample_rate,
                strategy="loudness",
                loudness_compressor=True
            )
            
            # Save metadata
            metadata = {
                'prompt': prompt,
                'duration': duration,
                'model': self.model_size,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'generated_at': timestamp,
                'style_profile_used': bool(style_profile)
            }
            
            metadata_file = output_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Generated audio saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _fallback_generate(
        self,
        prompt: Optional[str],
        duration: int,
        output_dir: str
    ) -> str:
        """
        Fallback generation when MusicGen is not available.
        Creates a simple synthesized tone as a placeholder.
        
        Args:
            prompt: Text prompt (ignored in fallback)
            duration: Duration in seconds
            output_dir: Output directory
            
        Returns:
            Path to generated audio file
        """
        print("MusicGen not available, generating placeholder audio...")
        
        import soundfile as sf
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate a simple harmonic tone
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a chord (major triad)
        freq_base = 440  # A4
        audio = (
            np.sin(2 * np.pi * freq_base * t) +
            0.5 * np.sin(2 * np.pi * freq_base * 1.25 * t) +  # Major third
            0.3 * np.sin(2 * np.pi * freq_base * 1.5 * t)     # Perfect fifth
        )
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.3
        
        # Simple envelope
        fade_len = int(0.1 * sample_rate)
        audio[:fade_len] *= np.linspace(0, 1, fade_len)
        audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"placeholder_{timestamp}.wav"
        output_file = output_path / filename
        
        sf.write(output_file, audio, sample_rate)
        
        print(f"Placeholder audio saved to: {output_file}")
        print("Note: Install audiocraft for AI music generation")
        
        return str(output_file)
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 3,
        duration: int = 30,
        output_dir: str = "./output/generated"
    ) -> list[str]:
        """
        Generate multiple variations of a prompt.
        
        Args:
            base_prompt: Base text prompt
            num_variations: Number of variations to generate
            duration: Duration in seconds for each
            output_dir: Output directory
            
        Returns:
            List of paths to generated files
        """
        variations = []
        
        for i in range(num_variations):
            print(f"\nGenerating variation {i+1}/{num_variations}...")
            
            # Vary temperature for different results
            temp = 1.0 + (i * 0.2)
            
            output_file = self.generate(
                prompt=base_prompt,
                duration=duration,
                output_dir=output_dir,
                temperature=temp
            )
            
            variations.append(output_file)
        
        return variations
    
    def continue_music(
        self,
        audio_path: str,
        continuation_duration: int = 15,
        output_dir: str = "./output/generated"
    ) -> str:
        """
        Continue an existing audio file (if model supports it).
        
        Args:
            audio_path: Path to audio file to continue
            continuation_duration: How long to continue (seconds)
            output_dir: Output directory
            
        Returns:
            Path to generated continuation
        """
        if self.model is None:
            print("Model not loaded, cannot continue audio")
            return ""
        
        # Note: This requires MusicGen melody model
        if self.model_size != "melody":
            print("Audio continuation requires MusicGen 'melody' model")
            return ""
        
        try:
            import torchaudio
            
            # Load the audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.model.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.model.sample_rate)
                audio = resampler(audio)
            
            # Generate continuation
            self.model.set_generation_params(duration=continuation_duration)
            wav = self.model.generate_continuation(
                audio.to(self.device),
                self.model.sample_rate,
                progress=True
            )
            
            # Save
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"continuation_{timestamp}.wav"
            output_file = output_path / filename
            
            from audiocraft.data.audio import audio_write
            audio_data = wav.squeeze(0).cpu().numpy()
            
            audio_write(
                str(output_file.with_suffix('')),
                audio_data,
                self.model.sample_rate,
                strategy="loudness"
            )
            
            print(f"Continuation saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"Error generating continuation: {e}")
            return ""

    def generate_section(
        self,
        section_type: str,
        style_profile: Optional[Dict[str, Any]] = None,
        artist_name: Optional[str] = None,
        duration: Optional[int] = None,
        output_dir: str = "./output/generated"
    ) -> str:
        """
        Generate audio for a specific song section.
        
        Args:
            section_type: Type of section (intro, verse, chorus, etc.)
            style_profile: Style profile to guide generation
            artist_name: Artist name to mimic
            duration: Duration in seconds (uses template default if None)
            output_dir: Output directory
            
        Returns:
            Path to generated audio file
        """
        section_info = self.SECTION_TEMPLATES.get(section_type, self.SECTION_TEMPLATES['verse'])
        
        if duration is None:
            duration = section_info['duration']
        
        # Build base prompt from style
        base_prompt = self.style_to_prompt(style_profile, artist_name)
        
        # Build section-specific prompt
        section_prompt = self.build_section_prompt(
            base_prompt,
            section_type,
            section_info['energy_modifier']
        )
        
        print(f"Generating {section_type} section ({duration}s)...")
        print(f"Section prompt: {section_prompt[:100]}...")
        
        return self.generate(
            prompt=section_prompt,
            duration=duration,
            style_profile=style_profile,
            artist_name=artist_name,
            output_dir=output_dir
        )

    def generate_from_blueprint(
        self,
        blueprint: Dict[str, Any],
        style_profile: Optional[Dict[str, Any]] = None,
        artist_name: Optional[str] = None,
        output_dir: str = "./output/generated"
    ) -> Dict[str, Any]:
        """
        Generate a full song from a blueprint with multiple sections.
        
        Args:
            blueprint: Song blueprint with structure and section info
            style_profile: Style profile to guide generation
            artist_name: Artist name to mimic
            output_dir: Output directory
            
        Returns:
            Dictionary with paths to generated sections and combined file
        """
        import soundfile as sf
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        sections = blueprint.get('sections', [])
        if not sections:
            sections = [
                {'type': 'intro', 'duration': 8},
                {'type': 'verse', 'duration': 20},
                {'type': 'chorus', 'duration': 16},
                {'type': 'verse', 'duration': 20},
                {'type': 'chorus', 'duration': 16},
                {'type': 'outro', 'duration': 10}
            ]
        
        generated_sections = []
        all_audio = []
        sample_rate = None
        
        for i, section in enumerate(sections):
            section_type = section.get('type', 'verse')
            duration = section.get('duration', self.SECTION_TEMPLATES.get(section_type, {}).get('duration', 16))
            
            print(f"\n[{i+1}/{len(sections)}] Generating {section_type}...")
            
            section_file = self.generate_section(
                section_type=section_type,
                style_profile=style_profile,
                artist_name=artist_name,
                duration=duration,
                output_dir=str(output_path / "sections")
            )
            
            generated_sections.append({
                'type': section_type,
                'duration': duration,
                'file': section_file
            })
            
            # Load audio for concatenation
            try:
                audio, sr = sf.read(section_file)
                if sample_rate is None:
                    sample_rate = sr
                all_audio.append(audio)
            except Exception as e:
                print(f"Warning: Could not load section audio: {e}")
        
        # Concatenate all sections
        combined_file = None
        if all_audio and sample_rate:
            try:
                # Simple concatenation with crossfade
                combined = self._concatenate_with_crossfade(all_audio, sample_rate)
                
                combined_filename = f"full_song_{timestamp}.wav"
                combined_file = output_path / combined_filename
                
                sf.write(combined_file, combined, sample_rate)
                print(f"\n✅ Full song saved to: {combined_file}")
            except Exception as e:
                print(f"Warning: Could not concatenate sections: {e}")
        
        # Save blueprint results
        result = {
            'blueprint': blueprint,
            'sections': generated_sections,
            'combined_file': str(combined_file) if combined_file else None,
            'timestamp': timestamp
        }
        
        result_file = output_path / f"blueprint_result_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result

    def _concatenate_with_crossfade(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int,
        crossfade_duration: float = 0.5
    ) -> np.ndarray:
        """
        Concatenate audio segments with crossfade.
        
        Args:
            audio_segments: List of audio arrays
            sample_rate: Sample rate
            crossfade_duration: Crossfade duration in seconds
            
        Returns:
            Combined audio array
        """
        if not audio_segments:
            return np.array([])
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        crossfade_samples = int(crossfade_duration * sample_rate)
        
        # Start with first segment
        result = audio_segments[0].copy()
        
        for segment in audio_segments[1:]:
            if len(result) < crossfade_samples or len(segment) < crossfade_samples:
                # Too short for crossfade, just concatenate
                result = np.concatenate([result, segment])
            else:
                # Apply crossfade
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                # Handle stereo vs mono
                if len(result.shape) > 1:
                    fade_out = fade_out.reshape(-1, 1)
                    fade_in = fade_in.reshape(-1, 1)
                
                # Crossfade region
                result[-crossfade_samples:] *= fade_out
                segment_copy = segment.copy()
                segment_copy[:crossfade_samples] *= fade_in
                
                # Overlap-add
                result[-crossfade_samples:] += segment_copy[:crossfade_samples]
                result = np.concatenate([result, segment_copy[crossfade_samples:]])
        
        return result

    def create_default_blueprint(
        self,
        total_duration: int = 120,
        style: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Create a default song blueprint.
        
        Args:
            total_duration: Approximate total duration in seconds
            style: Blueprint style ('standard', 'extended', 'short', 'trap', 'edm')
            
        Returns:
            Blueprint dictionary
        """
        blueprints = {
            'standard': [
                {'type': 'intro', 'duration': 8},
                {'type': 'verse', 'duration': 20},
                {'type': 'chorus', 'duration': 16},
                {'type': 'verse', 'duration': 20},
                {'type': 'chorus', 'duration': 16},
                {'type': 'bridge', 'duration': 12},
                {'type': 'chorus', 'duration': 16},
                {'type': 'outro', 'duration': 10}
            ],
            'short': [
                {'type': 'intro', 'duration': 4},
                {'type': 'verse', 'duration': 16},
                {'type': 'chorus', 'duration': 12},
                {'type': 'outro', 'duration': 8}
            ],
            'trap': [
                {'type': 'intro', 'duration': 8},
                {'type': 'hook', 'duration': 8},
                {'type': 'verse', 'duration': 24},
                {'type': 'hook', 'duration': 8},
                {'type': 'verse', 'duration': 24},
                {'type': 'hook', 'duration': 8},
                {'type': 'bridge', 'duration': 8},
                {'type': 'drop', 'duration': 16},
                {'type': 'outro', 'duration': 8}
            ],
            'edm': [
                {'type': 'intro', 'duration': 16},
                {'type': 'pre_chorus', 'duration': 16},
                {'type': 'drop', 'duration': 16},
                {'type': 'bridge', 'duration': 8},
                {'type': 'pre_chorus', 'duration': 16},
                {'type': 'drop', 'duration': 16},
                {'type': 'outro', 'duration': 16}
            ],
            'extended': [
                {'type': 'intro', 'duration': 12},
                {'type': 'verse', 'duration': 24},
                {'type': 'pre_chorus', 'duration': 8},
                {'type': 'chorus', 'duration': 20},
                {'type': 'verse', 'duration': 24},
                {'type': 'pre_chorus', 'duration': 8},
                {'type': 'chorus', 'duration': 20},
                {'type': 'bridge', 'duration': 16},
                {'type': 'chorus', 'duration': 20},
                {'type': 'outro', 'duration': 16}
            ]
        }
        
        sections = blueprints.get(style, blueprints['standard'])
        
        # Scale durations to match total_duration
        current_total = sum(s['duration'] for s in sections)
        scale_factor = total_duration / current_total
        
        scaled_sections = []
        for section in sections:
            scaled_sections.append({
                'type': section['type'],
                'duration': max(4, int(section['duration'] * scale_factor))
            })
        
        return {
            'style': style,
            'target_duration': total_duration,
            'sections': scaled_sections
        }
