"""
Music Generator Service
=======================
Generates music using Meta's AudioCraft (MusicGen).
Can be guided by style profiles and text prompts.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class MusicGenerator:
    """Generates music using AudioCraft's MusicGen."""
    
    def __init__(self, model_size: str = "small"):
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
            
            # Load on CPU first to avoid autocast issues during initialization on MPS
            if self.device == "mps":
                self.model = MusicGen.get_pretrained(self.model_size, device="cpu")
                print("Moving model to MPS...")
                self.model.to("mps")
            else:
                self.model = MusicGen.get_pretrained(self.model_size, device=self.device)
                
            print("MusicGen model loaded successfully")
        except ImportError:
            print("AudioCraft not available. Install with: pip install audiocraft")
            self.model = None
        except Exception as e:
            print(f"Error loading MusicGen model: {e}")
            self.model = None
    
    def style_to_prompt(self, style_profile: Optional[Dict[str, Any]] = None, artist_name: Optional[str] = None) -> str:
        """
        Convert a style profile to a text prompt for MusicGen.
        
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
        
        # Build descriptive prompt
        prompt_parts = []
        
        # Add artist context if known
        if artist_name:
            prompt_parts.append(f"in the style of {artist_name}")
            
            # Special artist-specific enhancers
            artist_lower = artist_name.lower()
            if 'yeat' in artist_lower:
                prompt_parts.extend(["aggressive trap beat", "heavy distorted 808", "synthetic bells", "rage trap"])
            elif 'playboi carti' in artist_lower or 'carti' in artist_lower:
                prompt_parts.extend(["minimalist trap", "high pitched synths", "vampire aesthetic", "f1lthy style"])
            elif 'travis scott' in artist_lower:
                prompt_parts.extend(["psychedelic trap", "dark atmospheric", "heavy reverb", "distorted vocals"])
        
        # Add overall characteristics
        if chars.get('overall'):
            prompt_parts.append(chars['overall'])
        
        # Add tempo info
        tempo_mean = tempo.get('mean', 120)
        prompt_parts.append(f"{int(tempo_mean)} BPM")
        
        # Add key if available
        most_common_key = key_info.get('most_common', '')
        if most_common_key:
            prompt_parts.append(f"in {most_common_key}")
        
        # Add energy descriptor
        if chars.get('energy'):
            prompt_parts.append(chars['energy'])

        # Add lyrical themes / keywords
        keywords = lyrics_info.get('top_keywords', [])
        if keywords:
            theme_str = ", ".join(keywords[:5])
            prompt_parts.append(f"themes of {theme_str}")
        
        prompt = ", ".join(prompt_parts) if prompt_parts else "pop music"
        
        return prompt
    
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
            
            # On MPS, we sometimes need to disable autocast to avoid "device_type" errors
            if self.device == "mps":
                import contextlib
                # Create a fake autocast to bypass internal ones if they fail
                # or just use a nullcontext if the user is on an older torch version
                try:
                    wav = self.model.generate(descriptions, progress=True)
                except Exception as e:
                    if "autocast" in str(e).lower():
                        print("MPS Autocast error detected, trying with CPU fallback for generation step...")
                        # Move to CPU for generation call if MPS fails specifically due to autocast
                        self.model.to("cpu")
                        wav = self.model.generate(descriptions, progress=True)
                        self.model.to("mps")
                    else:
                        raise e
            else:
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
