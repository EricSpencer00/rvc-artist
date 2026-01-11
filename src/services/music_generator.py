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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the MusicGen model."""
        try:
            from audiocraft.models import MusicGen
            print(f"Loading MusicGen model: {self.model_size} on {self.device}")
            self.model = MusicGen.get_pretrained(self.model_size, device=self.device)
            print("MusicGen model loaded successfully")
        except ImportError:
            print("AudioCraft not available. Install with: pip install audiocraft")
            self.model = None
        except Exception as e:
            print(f"Error loading MusicGen model: {e}")
            self.model = None
    
    def style_to_prompt(self, style_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a style profile to a text prompt for MusicGen.
        
        Args:
            style_profile: Style profile from StyleAnalyzer
            
        Returns:
            Text prompt describing the musical style
        """
        if not style_profile or not style_profile.get('characteristics'):
            return "pop music"
        
        chars = style_profile.get('characteristics', {})
        tempo = style_profile.get('tempo', {})
        key_info = style_profile.get('key', {})
        
        # Build descriptive prompt
        prompt_parts = []
        
        # Add overall characteristics
        if chars.get('overall'):
            prompt_parts.append(chars['overall'])
        
        # Add tempo info
        tempo_mean = tempo.get('mean', 120)
        if tempo_mean < 90:
            prompt_parts.append("slow tempo")
        elif tempo_mean > 140:
            prompt_parts.append("fast tempo")
        
        # Add key if available
        most_common_key = key_info.get('most_common', '')
        if most_common_key and 'minor' in most_common_key:
            prompt_parts.append("minor key")
        elif most_common_key and 'major' in most_common_key:
            prompt_parts.append("major key")
        
        # Add energy descriptor
        if chars.get('energy'):
            prompt_parts.append(chars['energy'])
        
        prompt = ", ".join(prompt_parts) if prompt_parts else "pop music"
        
        return prompt
    
    def generate(
        self,
        prompt: Optional[str] = None,
        duration: int = 30,
        style_profile: Optional[Dict[str, Any]] = None,
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
            output_dir: Directory to save generated audio
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Path to generated audio file
        """
        if self.model is None:
            return self._fallback_generate(prompt, duration, output_dir)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build the prompt
        if not prompt and style_profile:
            prompt = self.style_to_prompt(style_profile)
        elif not prompt:
            prompt = "pop music"
        
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
