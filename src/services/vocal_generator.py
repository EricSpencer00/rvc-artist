"""
Vocal Generator Service
=======================
Generates singing vocals with lyrics using various TTS/singing synthesis models.
Supports: Bark (Suno's TTS), MusicGen Melody with conditioning, RVC voice conversion.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


class VocalGenerator:
    """Generates singing vocals from lyrics and style guidance."""
    
    VOCAL_STYLES = {
        'rap': {
            'tempo_modifier': 1.0,
            'pitch_variance': 'low',
            'delivery': 'rhythmic, spoken-word style'
        },
        'singing': {
            'tempo_modifier': 0.9,
            'pitch_variance': 'medium',
            'delivery': 'melodic, sustained notes'
        },
        'melodic_rap': {
            'tempo_modifier': 0.95,
            'pitch_variance': 'medium',
            'delivery': 'melodic rap, sung-rap hybrid'
        },
        'aggressive': {
            'tempo_modifier': 1.1,
            'pitch_variance': 'high',
            'delivery': 'aggressive, intense, shouted'
        },
        'smooth': {
            'tempo_modifier': 0.85,
            'pitch_variance': 'low',
            'delivery': 'smooth, relaxed, flowing'
        }
    }
    
    def __init__(
        self,
        model_type: str = "bark",
        device: Optional[str] = None
    ):
        """
        Initialize vocal generator.
        
        Args:
            model_type: Type of model ('bark', 'musicgen_melody', 'rvc')
            device: Device to use (auto-detected if None)
        """
        self.model_type = model_type
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the vocal synthesis model."""
        print(f"Loading {self.model_type} vocal model on {self.device}...")
        
        if self.model_type == "bark":
            self._load_bark()
        elif self.model_type == "musicgen_melody":
            self._load_musicgen_melody()
        elif self.model_type == "rvc":
            print("RVC model requires separate setup - see RVC integration guide")
        else:
            print(f"Unknown model type: {self.model_type}")
    
    def _load_bark(self):
        """Load Bark TTS model (can generate singing)."""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            
            print("Preloading Bark models...")
            # This downloads models on first run
            preload_models(
                text_use_gpu=True if self.device in ["cuda", "mps"] else False,
                coarse_use_gpu=True if self.device in ["cuda", "mps"] else False,
                fine_use_gpu=True if self.device in ["cuda", "mps"] else False
            )
            
            self.model = {
                'generate': generate_audio,
                'sample_rate': SAMPLE_RATE
            }
            
            print(f"✅ Bark model loaded successfully")
            
        except ImportError:
            print("Bark not installed. Install with: pip install git+https://github.com/suno-ai/bark.git")
            self.model = None
        except Exception as e:
            print(f"Error loading Bark: {e}")
            self.model = None
    
    def _load_musicgen_melody(self):
        """Load MusicGen Melody model for vocal-like generation."""
        try:
            from audiocraft.models import MusicGen
            
            # MusicGen Melody is better for melodic content
            if self.device == "mps":
                # MPS has issues with MusicGen
                self.model = MusicGen.get_pretrained('melody', device="cpu")
                self.device = "cpu"
            else:
                self.model = MusicGen.get_pretrained('melody', device=self.device)
            
            print(f"✅ MusicGen Melody loaded on {self.device}")
            
        except ImportError:
            print("AudioCraft not installed. Install with: pip install audiocraft")
            self.model = None
        except Exception as e:
            print(f"Error loading MusicGen Melody: {e}")
            self.model = None
    
    def _format_lyrics_for_bark(
        self,
        lyrics: str,
        style: str = "rap",
        tempo_bpm: Optional[float] = None
    ) -> str:
        """
        Format lyrics for Bark with special tokens for singing/rap.
        
        Bark supports special syntax:
        - [laughter]
        - [music]
        - CAPITALS for emphasis
        - ... for pauses
        
        Args:
            lyrics: Raw lyrics text
            style: Vocal style (rap, singing, melodic_rap, etc.)
            tempo_bpm: Optional tempo for timing
            
        Returns:
            Formatted text for Bark
        """
        style_info = self.VOCAL_STYLES.get(style, self.VOCAL_STYLES['rap'])
        
        # Clean lyrics
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Add music cue at start
        formatted = "[music] "
        
        # Process each line
        for i, line in enumerate(lines):
            # Remove section markers like [VERSE], [CHORUS]
            if line.startswith('[') and line.endswith(']'):
                continue
            
            # Add pauses between lines
            if i > 0:
                formatted += "... "
            
            # For rap, add rhythmic markers
            if 'rap' in style:
                # Emphasize certain words (simple heuristic)
                words = line.split()
                processed_words = []
                for j, word in enumerate(words):
                    # Emphasize every 4th word (on beat)
                    if j % 4 == 0 and len(word) > 3:
                        processed_words.append(word.upper())
                    else:
                        processed_words.append(word)
                formatted += " ".join(processed_words) + " "
            else:
                # For singing, keep it more natural
                formatted += line + " "
        
        # Add music outro
        formatted += " [music]"
        
        return formatted
    
    def generate_with_bark(
        self,
        lyrics: str,
        style: str = "rap",
        voice_preset: str = "v2/en_speaker_6",  # Male rap voice
        tempo_bpm: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate vocals using Bark.
        
        Args:
            lyrics: Lyrics to synthesize
            style: Vocal style
            voice_preset: Bark voice preset
            tempo_bpm: Optional tempo
            
        Returns:
            Audio array
        """
        if self.model is None:
            print("Bark model not available")
            return np.zeros(48000 * 5)  # 5 seconds of silence
        
        # Format lyrics for Bark
        formatted_text = self._format_lyrics_for_bark(lyrics, style, tempo_bpm)
        
        print(f"🎤 Generating vocals with Bark...")
        print(f"   Style: {style}")
        print(f"   Voice: {voice_preset}")
        print(f"   Text preview: {formatted_text[:100]}...")
        
        try:
            # Generate audio
            audio_array = self.model['generate'](
                formatted_text,
                history_prompt=voice_preset
            )
            
            print(f"✅ Generated {len(audio_array) / self.model['sample_rate']:.1f}s of vocals")
            
            return audio_array
            
        except Exception as e:
            print(f"Error generating with Bark: {e}")
            return np.zeros(48000 * 5)
    
    def generate_with_musicgen(
        self,
        lyrics: str,
        style_prompt: str,
        duration: int = 15,
        melody_reference: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate vocal-like audio using MusicGen Melody.
        
        Note: MusicGen doesn't directly synthesize lyrics, but can create
        vocal-like timbres and melodies.
        
        Args:
            lyrics: Lyrics (used for style guidance only)
            style_prompt: Musical style prompt
            duration: Duration in seconds
            melody_reference: Optional melody to condition on
            temperature: Sampling temperature
            
        Returns:
            Audio array
        """
        if self.model is None:
            print("MusicGen Melody model not available")
            return np.zeros(32000 * duration)
        
        # Build prompt emphasizing vocals
        prompt = f"{style_prompt}, human voice, vocal melody, singing, acapella style"
        
        print(f"🎤 Generating vocal-style audio with MusicGen Melody...")
        print(f"   Prompt: {prompt}")
        
        try:
            self.model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_k=250
            )
            
            # If melody reference provided, use it for conditioning
            if melody_reference is not None:
                # MusicGen Melody can condition on a melody
                # Convert reference to proper format
                if isinstance(melody_reference, np.ndarray):
                    melody_reference = torch.from_numpy(melody_reference).float()
                
                if melody_reference.dim() == 1:
                    melody_reference = melody_reference.unsqueeze(0).unsqueeze(0)
                elif melody_reference.dim() == 2:
                    melody_reference = melody_reference.unsqueeze(0)
                
                wav = self.model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody_reference,
                    melody_sample_rate=32000,
                    progress=False
                )
            else:
                wav = self.model.generate(
                    descriptions=[prompt],
                    progress=False
                )
            
            audio = wav.squeeze(0).cpu().numpy()
            
            print(f"✅ Generated {duration}s of vocal-style audio")
            
            return audio
            
        except Exception as e:
            print(f"Error generating with MusicGen: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(32000 * duration)
    
    def generate(
        self,
        lyrics: str,
        style: str = "rap",
        duration: int = 15,
        style_profile: Optional[Dict] = None,
        artist_name: Optional[str] = None,
        voice_preset: Optional[str] = None,
        melody_reference: Optional[np.ndarray] = None,
        output_dir: str = "./output/vocals",
        save_audio: bool = True
    ) -> str:
        """
        Generate vocals from lyrics.
        
        Args:
            lyrics: Lyrics text
            style: Vocal style (rap, singing, melodic_rap, etc.)
            duration: Target duration
            style_profile: Optional style profile
            artist_name: Artist name for style
            voice_preset: Voice preset (for Bark)
            melody_reference: Optional melody to follow
            output_dir: Output directory
            save_audio: Whether to save audio file
            
        Returns:
            Path to generated audio (if saved) or empty string
        """
        # Select default voice based on artist
        if voice_preset is None:
            # Bark voice presets (examples)
            voice_map = {
                'yeat': 'v2/en_speaker_6',  # Young male, energetic
                'playboi carti': 'v2/en_speaker_9',  # High-pitched male
                'travis scott': 'v2/en_speaker_6',  # Deep male
                'drake': 'v2/en_speaker_3',  # Smooth male
                'juice wrld': 'v2/en_speaker_7'  # Melodic male
            }
            voice_preset = voice_map.get(
                artist_name.lower() if artist_name else '',
                'v2/en_speaker_6'
            )
        
        # Generate based on model type
        if self.model_type == "bark":
            audio = self.generate_with_bark(
                lyrics=lyrics,
                style=style,
                voice_preset=voice_preset
            )
            sample_rate = 24000  # Bark's sample rate
            
        elif self.model_type == "musicgen_melody":
            # Build style prompt
            if style_profile:
                tempo = style_profile.get('tempo', {}).get('mean', 140)
                key = style_profile.get('key', {}).get('most_common', 'C major')
                style_prompt = f"{artist_name} style, {tempo} BPM, {key}, vocal melody"
            else:
                style_prompt = f"{artist_name} style" if artist_name else "vocal melody"
            
            audio = self.generate_with_musicgen(
                lyrics=lyrics,
                style_prompt=style_prompt,
                duration=duration,
                melody_reference=melody_reference
            )
            sample_rate = 32000  # MusicGen sample rate
        
        else:
            print(f"Unsupported model type: {self.model_type}")
            return ""
        
        # Save if requested
        if save_audio:
            import soundfile as sf
            from datetime import datetime
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = output_path / f"vocals_{style}_{timestamp}.wav"
            
            # Ensure audio is in correct format
            if audio.ndim == 2:
                audio_to_save = audio.T
            else:
                audio_to_save = audio
            
            sf.write(audio_file, audio_to_save, sample_rate)
            
            # Save metadata
            metadata = {
                'lyrics': lyrics,
                'style': style,
                'model_type': self.model_type,
                'voice_preset': voice_preset,
                'artist_name': artist_name,
                'duration': len(audio) / sample_rate,
                'generated_at': timestamp
            }
            
            metadata_file = audio_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Vocals saved: {audio_file}")
            
            return str(audio_file)
        
        return ""
    
    def list_bark_voices(self) -> List[str]:
        """List available Bark voice presets."""
        # Common Bark v2 voice presets
        voices = [
            "v2/en_speaker_0",  # Male 1
            "v2/en_speaker_1",  # Male 2
            "v2/en_speaker_2",  # Male 3
            "v2/en_speaker_3",  # Male 4 (smooth)
            "v2/en_speaker_4",  # Male 5
            "v2/en_speaker_5",  # Male 6
            "v2/en_speaker_6",  # Male 7 (energetic)
            "v2/en_speaker_7",  # Male 8 (young)
            "v2/en_speaker_8",  # Male 9
            "v2/en_speaker_9",  # Male 10 (high pitch)
        ]
        
        return voices


if __name__ == "__main__":
    # Test vocal generation
    print("Testing Vocal Generator...")
    
    # Try Bark first (best for actual singing/rapping)
    try:
        generator = VocalGenerator(model_type="bark")
        
        test_lyrics = """
        Yeah, I'm ballin' like I'm Jordan
        Money comin' in, I can't ignore it
        Pull up in the foreign, doors is soarin'
        They be hatin' but I keep on scorin'
        """
        
        vocal_path = generator.generate(
            lyrics=test_lyrics,
            style="rap",
            artist_name="Yeat",
            duration=10
        )
        
        print(f"\n✅ Bark test complete: {vocal_path}")
        
    except Exception as e:
        print(f"Bark test failed: {e}")
        print("Trying MusicGen Melody fallback...")
        
        generator = VocalGenerator(model_type="musicgen_melody")
        
        vocal_path = generator.generate(
            lyrics=test_lyrics,
            style="rap",
            artist_name="Yeat",
            duration=10
        )
        
        print(f"\n✅ MusicGen test complete: {vocal_path}")
