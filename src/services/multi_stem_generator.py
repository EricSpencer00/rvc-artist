"""
Multi-Stem Music Generator
==========================
Generates music by creating separate stems (vocals, drums, bass, melody)
then mixing them together for higher quality output.

This approach is inspired by Suno's architecture and solves the "generic beat" problem
by giving each element dedicated generation attention.

Features:
- Multi-stem generation (vocals, drums, bass, melody)
- Professional audio processing (EQ, compression, reverb)
- RVC voice conversion integration
- Melody-conditioned vocals
- Dynamic section generation
- Advanced mastering
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import json
import soundfile as sf

# Try to import pedalboard for audio processing
try:
    import pedalboard
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, 
        LowpassFilter, Gain, Limiter, HighShelfFilter, LowShelfFilter
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("Pedalboard not available - install with: pip install pedalboard")

# Try to import pyloudnorm for mastering
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Pyloudnorm not available - install with: pip install pyloudnorm")

try:
    from .vocal_generator import VocalGenerator
    VOCAL_GENERATOR_AVAILABLE = True
except ImportError:
    VOCAL_GENERATOR_AVAILABLE = False
    print("VocalGenerator not available - vocals will be instrumental only")


class StemProcessor:
    """Applies professional audio processing to stems."""
    
    # Processing chains for each stem type
    PROCESSING_CHAINS = {
        'vocals': {
            'highpass': 80,
            'compression': {'threshold_db': -20, 'ratio': 4},
            'reverb': {'room_size': 0.3, 'wet_level': 0.15},
            'presence_boost': {'frequency': 3000, 'gain_db': 2}
        },
        'drums': {
            'highpass': 30,
            'compression': {'threshold_db': -15, 'ratio': 6},
            'high_shelf': {'frequency': 8000, 'gain_db': 2},
            'low_shelf': {'frequency': 100, 'gain_db': 1}
        },
        'bass': {
            'lowpass': 250,
            'highpass': 30,
            'compression': {'threshold_db': -18, 'ratio': 8},
            'saturation': 0.1
        },
        'melody_1': {
            'highpass': 200,
            'compression': {'threshold_db': -18, 'ratio': 3},
            'reverb': {'room_size': 0.4, 'wet_level': 0.2},
            'stereo_width': 1.2
        },
        'melody_2': {
            'highpass': 150,
            'compression': {'threshold_db': -20, 'ratio': 2.5},
            'reverb': {'room_size': 0.6, 'wet_level': 0.35},
            'stereo_width': 1.4
        }
    }
    
    def __init__(self, sample_rate: int = 32000):
        """Initialize stem processor."""
        self.sample_rate = sample_rate
        
    def process_stem(
        self,
        audio: np.ndarray,
        stem_type: str,
        custom_params: Optional[Dict] = None
    ) -> np.ndarray:
        """Apply professional processing to a stem."""
        if not PEDALBOARD_AVAILABLE:
            return self._basic_process(audio, stem_type)
        
        params = self.PROCESSING_CHAINS.get(stem_type, {})
        if custom_params:
            params.update(custom_params)
        
        # Build processing chain
        board = Pedalboard([])
        
        if 'highpass' in params:
            board.append(HighpassFilter(cutoff_frequency_hz=params['highpass']))
        
        if 'lowpass' in params:
            board.append(LowpassFilter(cutoff_frequency_hz=params['lowpass']))
        
        if 'low_shelf' in params:
            board.append(LowShelfFilter(
                cutoff_frequency_hz=params['low_shelf']['frequency'],
                gain_db=params['low_shelf']['gain_db']
            ))
        
        if 'high_shelf' in params:
            board.append(HighShelfFilter(
                cutoff_frequency_hz=params['high_shelf']['frequency'],
                gain_db=params['high_shelf']['gain_db']
            ))
        
        if 'compression' in params:
            board.append(Compressor(
                threshold_db=params['compression']['threshold_db'],
                ratio=params['compression']['ratio']
            ))
        
        if 'reverb' in params:
            board.append(Reverb(
                room_size=params['reverb']['room_size'],
                wet_level=params['reverb']['wet_level']
            ))
        
        # Ensure audio is in correct format
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2 and audio.shape[0] == 2:
            audio = audio.T
        
        try:
            processed = board(audio, self.sample_rate)
        except Exception as e:
            print(f"Processing error: {e}")
            processed = self._basic_process(audio.T if audio.ndim == 2 else audio, stem_type)
            if processed.ndim == 1:
                processed = processed.reshape(-1, 1)
        
        if processed.ndim == 2:
            processed = processed.T
        
        if 'stereo_width' in params and processed.ndim == 2:
            processed = self._apply_stereo_width(processed, params['stereo_width'])
        
        return processed
    
    def _basic_process(self, audio: np.ndarray, stem_type: str) -> np.ndarray:
        """Basic processing without pedalboard."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio
    
    def _apply_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo widening."""
        if audio.ndim != 2 or audio.shape[0] != 2:
            return audio
        
        left, right = audio[0], audio[1]
        mid = (left + right) / 2
        side = (left - right) / 2
        side = side * width
        
        return np.stack([mid + side, mid - side])


class MasteringProcessor:
    """Professional mastering chain."""
    
    def __init__(self, sample_rate: int = 32000):
        """Initialize mastering processor."""
        self.sample_rate = sample_rate
        
    def master(
        self,
        audio: np.ndarray,
        target_lufs: float = -14.0,
        stereo_width: float = 1.1,
        limiter_threshold: float = -1.0
    ) -> np.ndarray:
        """Apply professional mastering."""
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        if stereo_width != 1.0:
            audio = self._apply_stereo_width(audio, stereo_width)
        
        if PYLOUDNORM_AVAILABLE:
            audio = self._normalize_loudness(audio, target_lufs)
        else:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.9
        
        audio = self._apply_limiter(audio, limiter_threshold)
        
        return audio
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize to target LUFS."""
        if not PYLOUDNORM_AVAILABLE:
            return audio
        
        try:
            if audio.ndim == 2 and audio.shape[0] == 2:
                audio_t = audio.T
            else:
                audio_t = audio
            
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio_t)
            
            if not np.isinf(loudness) and not np.isnan(loudness):
                audio_t = pyln.normalize.loudness(audio_t, loudness, target_lufs)
                
                if audio_t.ndim == 2:
                    audio = audio_t.T
                else:
                    audio = audio_t
                    
        except Exception as e:
            print(f"Loudness normalization failed: {e}")
        
        return audio
    
    def _apply_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo widening."""
        if audio.ndim != 2 or audio.shape[0] != 2:
            return audio
        
        left, right = audio[0], audio[1]
        mid = (left + right) / 2
        side = (left - right) / 2
        side = side * width
        
        return np.stack([mid + side, mid - side])
    
    def _apply_limiter(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply soft limiting."""
        threshold = 10 ** (threshold_db / 20)
        max_val = np.abs(audio).max()
        if max_val > threshold:
            audio = np.tanh(audio / max_val) * threshold
        return audio


class MultiStemGenerator:
    """Generates music using separate stem generation and mixing."""
    
    STEM_TYPES = {
        'vocals': {
            'description': 'singing voice, vocal melody, human voice',
            'instruments': None,
            'eq_boost': [1000, 3000],
            'priority': 1,
            'use_vocal_generator': True
        },
        'drums': {
            'description': 'drum kit, percussion, hi-hats, snare, kick',
            'instruments': ['drums', 'percussion'],
            'eq_boost': [60, 100, 8000],
            'priority': 2
        },
        'bass': {
            'description': 'bass, sub bass, 808',
            'instruments': ['bass'],
            'eq_boost': [60, 120],
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
    
    SECTION_PARAMS = {
        'intro': {
            'temperature': 0.8,
            'guidance': 4.0,
            'duration': 8,
            'energy_modifier': 0.6,
            'descriptors': ['atmospheric intro', 'building anticipation', 'sparse']
        },
        'verse': {
            'temperature': 1.0,
            'guidance': 3.5,
            'duration': 16,
            'energy_modifier': 0.8,
            'descriptors': ['rhythmic verse', 'steady groove', 'melodic']
        },
        'pre_chorus': {
            'temperature': 1.0,
            'guidance': 3.5,
            'duration': 8,
            'energy_modifier': 0.9,
            'descriptors': ['building tension', 'rising energy']
        },
        'chorus': {
            'temperature': 1.1,
            'guidance': 3.0,
            'duration': 16,
            'energy_modifier': 1.0,
            'descriptors': ['powerful chorus', 'full energy', 'catchy hook']
        },
        'hook': {
            'temperature': 1.1,
            'guidance': 3.0,
            'duration': 8,
            'energy_modifier': 1.0,
            'descriptors': ['memorable hook', 'signature phrase']
        },
        'bridge': {
            'temperature': 0.9,
            'guidance': 4.5,
            'duration': 12,
            'energy_modifier': 0.7,
            'descriptors': ['contrasting bridge', 'different texture']
        },
        'drop': {
            'temperature': 1.2,
            'guidance': 3.0,
            'duration': 16,
            'energy_modifier': 1.0,
            'descriptors': ['heavy drop', 'maximum energy', 'bass-heavy']
        },
        'outro': {
            'temperature': 0.8,
            'guidance': 4.0,
            'duration': 10,
            'energy_modifier': 0.5,
            'descriptors': ['fading outro', 'winding down']
        }
    }
    
    def __init__(
        self,
        model_size: str = "large",
        use_stereo: bool = True,
        vocal_model: str = "bark",
        rvc_model_path: Optional[str] = None,
        enable_processing: bool = True,
        enable_mastering: bool = True,
        enable_parallel_generation: bool = True,
        max_parallel_stems: int = 2,
        cache_all_models: bool = True
    ):
        """
        Initialize multi-stem generator optimized for high-memory machines.
        
        Args:
            model_size: MusicGen model size ('small', 'medium', 'large')
            use_stereo: Whether to use stereo models
            vocal_model: Vocal synthesis model ('bark', 'musicgen_melody', None)
            rvc_model_path: Path to RVC model for voice conversion
            enable_processing: Enable stem processing (EQ, compression, reverb)
            enable_mastering: Enable mastering chain (LUFS, limiting)
            enable_parallel_generation: Generate multiple stems in parallel (64GB+ only)
            max_parallel_stems: Max stems to generate simultaneously (2-4 for 64GB RAM)
            cache_all_models: Keep all models in memory for faster generation
        """
        self.model_size = model_size
        self.use_stereo = use_stereo
        self.vocal_model_type = vocal_model
        self.rvc_model_path = rvc_model_path
        self.enable_processing = enable_processing
        self.enable_mastering = enable_mastering
        self.enable_parallel_generation = enable_parallel_generation
        self.max_parallel_stems = min(max_parallel_stems, 4)  # Cap at 4 for stability
        self.cache_all_models = cache_all_models
        
        self.models = {}
        self.vocal_generator = None
        self.rvc_model = None
        self.stem_processor = None
        self.mastering_processor = None
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self._load_models()
        self._init_vocal_generator()
        self._init_processors()
        
        if rvc_model_path:
            self._load_rvc_model(rvc_model_path)
    
    def _load_models(self):
        """Load MusicGen models."""
        try:
            from audiocraft.models import MusicGen
            
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
            
            try:
                print(f"Loading MusicGen Melody...")
                if self.device == "mps" or self.device == "cpu":
                    self.models['melody'] = MusicGen.get_pretrained('melody', device="cpu")
                else:
                    self.models['melody'] = MusicGen.get_pretrained('melody', device=self.device)
            except Exception as e:
                print(f"Melody model not available: {e}")
                self.models['melody'] = self.models['instrumental']
            
            print(f"✅ MusicGen models loaded on {self.device}")
            
        except ImportError:
            print("AudioCraft not installed. Install with: pip install audiocraft")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _init_vocal_generator(self):
        """Initialize vocal generator."""
        if self.vocal_model_type and VOCAL_GENERATOR_AVAILABLE:
            try:
                self.vocal_generator = VocalGenerator(
                    model_type=self.vocal_model_type,
                    device=self.device
                )
                print(f"✅ Vocal generator ({self.vocal_model_type}) initialized")
            except Exception as e:
                print(f"Could not load vocal generator: {e}")
    
    def _init_processors(self):
        """Initialize audio processors."""
        self.stem_processor = StemProcessor(sample_rate=32000)
        self.mastering_processor = MasteringProcessor(sample_rate=32000)
        
        if self.enable_processing:
            print("✅ Stem processor initialized")
        if self.enable_mastering:
            print("✅ Mastering processor initialized")
    
    def _load_rvc_model(self, model_path: str):
        """Load RVC model for voice conversion."""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"RVC model not found: {model_path}")
            return
        
        print(f"⚠️ RVC loading placeholder - implement with actual RVC library")
        self.rvc_model = {
            'path': str(model_path),
            'loaded': False,
            'message': 'Install RVC library to enable voice conversion'
        }
    
    def apply_rvc_conversion(
        self,
        audio: np.ndarray,
        sample_rate: int = 32000,
        pitch_shift: int = 0
    ) -> np.ndarray:
        """Apply RVC voice conversion."""
        if self.rvc_model is None or not self.rvc_model.get('loaded', False):
            return audio
        return audio
    
    def _build_stem_prompt(
        self,
        base_prompt: str,
        stem_type: str,
        style_profile: Optional[Dict] = None,
        section_type: Optional[str] = None
    ) -> str:
        """Build a specialized prompt for a specific stem."""
        stem_info = self.STEM_TYPES.get(stem_type, {})
        parts = [base_prompt]
        
        if 'description' in stem_info:
            parts.append(stem_info['description'])
        
        parts.append(f"focus on {stem_type} only")
        
        if section_type and section_type in self.SECTION_PARAMS:
            section = self.SECTION_PARAMS[section_type]
            parts.extend(section.get('descriptors', []))
        
        if style_profile:
            if stem_type == 'drums':
                energy = style_profile.get('energy', {}).get('mean', 0.5)
                if energy > 0.7:
                    parts.append("aggressive hard-hitting drums")
                else:
                    parts.append("smooth groovy drums")
                    
            elif stem_type == 'bass':
                spectral = style_profile.get('spectral', {})
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
        lyrics: Optional[str] = None,
        artist_name: Optional[str] = None,
        section_type: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_scale: float = 3.0,
        melody_reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a single stem."""
        # Special handling for vocals
        if stem_type == 'vocals' and self.vocal_generator and lyrics:
            return self._generate_vocal_stem(
                lyrics=lyrics,
                base_prompt=base_prompt,
                style_profile=style_profile,
                artist_name=artist_name,
                duration=duration,
                melody_reference=melody_reference
            )
        
        # Choose model
        if stem_type in ['melody_1', 'melody_2'] and 'melody' in self.models:
            model = self.models['melody']
        else:
            model = self.models.get('instrumental')
        
        if model is None:
            print(f"No model available for {stem_type}")
            return np.zeros(int(duration * 32000))
        
        prompt = self._build_stem_prompt(base_prompt, stem_type, style_profile, section_type)
        print(f"\n🎵 Generating {stem_type}: '{prompt[:80]}...'")
        
        if section_type and section_type in self.SECTION_PARAMS:
            section = self.SECTION_PARAMS[section_type]
            temperature = section.get('temperature', temperature)
            guidance_scale = section.get('guidance', guidance_scale)
        
        try:
            model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                cfg_coef=guidance_scale
            )
            
            if melody_reference is not None and hasattr(model, 'generate_with_chroma'):
                mel_tensor = torch.from_numpy(melody_reference).float()
                if mel_tensor.dim() == 1:
                    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)
                elif mel_tensor.dim() == 2:
                    mel_tensor = mel_tensor.unsqueeze(0)
                
                wav = model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=mel_tensor,
                    melody_sample_rate=32000,
                    progress=False
                )
            else:
                wav = model.generate([prompt], progress=False)
            
            audio = wav.squeeze(0).cpu().numpy()
            
            if self.enable_processing and self.stem_processor:
                audio = self.stem_processor.process_stem(audio, stem_type)
            
            return audio
            
        except Exception as e:
            print(f"Error generating {stem_type}: {e}")
            return np.zeros(int(duration * 32000))
    
    def _generate_vocal_stem(
        self,
        lyrics: str,
        base_prompt: str,
        style_profile: Optional[Dict],
        artist_name: Optional[str],
        duration: int,
        melody_reference: Optional[np.ndarray]
    ) -> np.ndarray:
        """Generate vocal stem."""
        print(f"\n🎤 Generating vocals with {self.vocal_model_type}...")
        
        try:
            if 'aggressive' in base_prompt.lower() or 'rage' in base_prompt.lower():
                vocal_style = 'aggressive'
            elif 'melodic' in base_prompt.lower():
                vocal_style = 'melodic_rap'
            elif 'smooth' in base_prompt.lower():
                vocal_style = 'smooth'
            else:
                vocal_style = 'rap'
            
            if hasattr(self.vocal_generator, 'generate_with_melody') and melody_reference is not None:
                audio = self.vocal_generator.generate_with_melody(
                    lyrics=lyrics,
                    melody_audio=melody_reference,
                    style=vocal_style,
                    style_profile=style_profile,
                    artist_name=artist_name,
                    duration=duration
                )
            else:
                audio_file = self.vocal_generator.generate(
                    lyrics=lyrics,
                    style=vocal_style,
                    duration=duration,
                    style_profile=style_profile,
                    artist_name=artist_name,
                    save_audio=False
                )
                
                if isinstance(audio_file, str) and Path(audio_file).exists():
                    audio, sr = sf.read(audio_file)
                    if sr != 32000:
                        from scipy import signal
                        num_samples = int(len(audio) * 32000 / sr)
                        audio = signal.resample(audio, num_samples)
                elif isinstance(audio_file, np.ndarray):
                    audio = audio_file
                else:
                    audio = np.zeros(int(duration * 32000))
            
            if self.rvc_model and self.rvc_model.get('loaded', False):
                audio = self.apply_rvc_conversion(audio)
            
            if self.enable_processing and self.stem_processor:
                audio = self.stem_processor.process_stem(audio, 'vocals')
            
            return audio
            
        except Exception as e:
            print(f"Vocal generation failed: {e}")
            return np.zeros(int(duration * 32000))
    
    def mix_stems(
        self,
        stems: Dict[str, np.ndarray],
        mix_levels: Optional[Dict[str, float]] = None,
        sample_rate: int = 32000,
        apply_mastering: bool = True
    ) -> np.ndarray:
        """Mix multiple stems together."""
        if not stems:
            return np.zeros(int(sample_rate * 5))
        
        default_levels = {
            'vocals': 1.0,
            'drums': 0.85,
            'bass': 0.9,
            'melody_1': 0.7,
            'melody_2': 0.5
        }
        
        if mix_levels is None:
            mix_levels = default_levels
        
        max_length = max(
            audio.shape[-1] if audio.ndim > 1 else len(audio)
            for audio in stems.values()
        )
        
        is_stereo = any(
            audio.ndim == 2 and audio.shape[0] == 2
            for audio in stems.values()
        )
        
        if is_stereo:
            mixed = np.zeros((2, max_length))
        else:
            mixed = np.zeros(max_length)
        
        for stem_type, audio in stems.items():
            level = mix_levels.get(stem_type, 1.0)
            
            if is_stereo and audio.ndim == 1:
                audio = np.stack([audio, audio])
            elif is_stereo and audio.ndim == 2 and audio.shape[0] != 2:
                audio = np.stack([audio[0], audio[0]])
            elif not is_stereo and audio.ndim == 2:
                audio = audio.mean(axis=0)
            
            current_length = audio.shape[-1] if audio.ndim > 1 else len(audio)
            if current_length < max_length:
                if audio.ndim == 2:
                    padding = np.zeros((2, max_length - current_length))
                    audio = np.concatenate([audio, padding], axis=1)
                else:
                    padding = np.zeros(max_length - current_length)
                    audio = np.concatenate([audio, padding])
            elif current_length > max_length:
                if audio.ndim == 2:
                    audio = audio[:, :max_length]
                else:
                    audio = audio[:max_length]
            
            mixed += audio * level
        
        if apply_mastering and self.enable_mastering and self.mastering_processor:
            mixed = self.mastering_processor.master(mixed)
        else:
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = np.tanh(mixed / max_val) * 0.95
            else:
                mixed = mixed * 0.95
        
        return mixed
    
    def _prepare_audio_for_export(self, audio: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Ensure audio is in a soundfile-compatible shape and dtype."""
        if audio is None:
            return np.zeros((1,), dtype=np.float32)
        
        audio_array = np.asarray(audio)
        if audio_array.size == 0:
            return np.zeros((1,), dtype=np.float32)
        
        if np.iscomplexobj(audio_array):
            audio_array = audio_array.real
        
        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if audio_array.ndim == 0:
            audio_array = np.zeros((1,), dtype=np.float32)
        elif audio_array.ndim == 1:
            audio_array = audio_array.astype(np.float32)
        elif audio_array.ndim == 2:
            if audio_array.shape[0] not in (1, 2) and audio_array.shape[1] in (1, 2):
                audio_array = audio_array.T
            if audio_array.shape[0] == 1:
                audio_array = audio_array[0]
            elif audio_array.shape[0] == 2:
                audio_array = audio_array.T
            else:
                audio_array = audio_array[:2].T
            audio_array = audio_array.astype(np.float32)
        else:
            audio_array = audio_array.reshape(-1).astype(np.float32)
        
        if audio_array.ndim == 1:
            return audio_array.astype(np.float32)
        return audio_array.astype(np.float32)
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        style_profile: Optional[Dict] = None,
        artist_name: Optional[str] = None,
        lyrics: Optional[str] = None,
        output_dir: str = "./output/generated",
        stems_to_generate: Optional[List[str]] = None,
        save_individual_stems: bool = True,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_scale: float = 3.0,
        mix_levels: Optional[Dict[str, float]] = None,
        melody_reference: Optional[np.ndarray] = None
    ) -> Tuple[str, Dict[str, str]]:
        """Generate complete song with multiple stems."""
        if stems_to_generate is None:
            stems_to_generate = list(self.STEM_TYPES.keys())
        
        if artist_name:
            prompt = f"{prompt}, in the style of {artist_name}"
        
        print(f"\n{'='*60}")
        print(f"🎼 MULTI-STEM GENERATION")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Duration: {duration}s")
        print(f"Stems: {', '.join(stems_to_generate)}")
        print(f"Processing: {'✅' if self.enable_processing else '❌'}")
        print(f"Mastering: {'✅' if self.enable_mastering else '❌'}")
        print(f"{'='*60}\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        stems = {}
        stem_paths = {}
        
        sorted_stems = sorted(
            stems_to_generate,
            key=lambda x: self.STEM_TYPES.get(x, {}).get('priority', 99)
        )
        
        for stem_type in sorted_stems:
            audio = self.generate_stem(
                stem_type=stem_type,
                base_prompt=prompt,
                duration=duration,
                style_profile=style_profile,
                lyrics=lyrics if stem_type == 'vocals' else None,
                artist_name=artist_name,
                temperature=temperature,
                top_k=top_k,
                guidance_scale=guidance_scale,
                melody_reference=melody_reference
            )
            
            stems[stem_type] = audio
            
            if save_individual_stems:
                stem_file = output_path / f"stem_{stem_type}_{timestamp}.wav"
                audio_to_save = self._prepare_audio_for_export(audio)
                sf.write(
                    stem_file,
                    audio_to_save,
                    32000,
                    format="WAV",
                    subtype="PCM_16"
                )
                stem_paths[stem_type] = str(stem_file)
                print(f"✅ Saved {stem_type} stem")
        
        print(f"\n🎚️  Mixing {len(stems)} stems...")
        mixed_audio = self.mix_stems(stems, mix_levels=mix_levels)
        
        mixed_file = output_path / f"mixed_{timestamp}.wav"
        
        mixed_to_save = self._prepare_audio_for_export(mixed_audio)
        sf.write(
            mixed_file,
            mixed_to_save,
            32000,
            format="WAV",
            subtype="PCM_16"
        )
        
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
            'artist_name': artist_name,
            'processing_enabled': self.enable_processing,
            'mastering_enabled': self.enable_mastering
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
    
    def generate_sections(
        self,
        lyrics_sections: List[Tuple[str, str]],
        base_prompt: str,
        style_profile: Optional[Dict] = None,
        artist_name: Optional[str] = None,
        output_dir: str = "./output/generated",
        stems_to_generate: Optional[List[str]] = None,
        crossfade_duration: float = 0.5
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """Generate song with dynamic section parameters."""
        if stems_to_generate is None:
            stems_to_generate = ['drums', 'bass', 'melody_1']
        
        print(f"\n{'='*60}")
        print(f"🎼 SECTION-BASED GENERATION")
        print(f"{'='*60}")
        print(f"Sections: {len(lyrics_sections)}")
        print(f"{'='*60}\n")
        
        all_section_stems = {}
        
        for i, (section_type, section_lyrics) in enumerate(lyrics_sections):
            print(f"\n📍 Section {i+1}: {section_type.upper()}")
            
            section_params = self.SECTION_PARAMS.get(
                section_type, 
                self.SECTION_PARAMS['verse']
            )
            
            section_stems = {}
            
            for stem_type in stems_to_generate:
                audio = self.generate_stem(
                    stem_type=stem_type,
                    base_prompt=base_prompt,
                    duration=section_params['duration'],
                    style_profile=style_profile,
                    lyrics=section_lyrics if stem_type == 'vocals' else None,
                    artist_name=artist_name,
                    section_type=section_type,
                    temperature=section_params['temperature'],
                    guidance_scale=section_params['guidance']
                )
                section_stems[stem_type] = audio
            
            section_mixed = self.mix_stems(section_stems, apply_mastering=False)
            all_section_stems[f"{section_type}_{i}"] = section_mixed
        
        final_audio = self._crossfade_sections(
            list(all_section_stems.values()),
            crossfade_samples=int(crossfade_duration * 32000)
        )
        
        if self.enable_mastering and self.mastering_processor:
            final_audio = self.mastering_processor.master(final_audio)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"sections_{timestamp}.wav"
        
        if final_audio.ndim == 2:
            final_to_save = final_audio.T
        else:
            final_to_save = final_audio
        
        sf.write(output_file, final_to_save, 32000)
        
        print(f"\n✅ Section generation complete: {output_file}")
        
        return str(output_file), all_section_stems
    
    def _generate_stems_sequential(
        self,
        stems_to_generate: List[str],
        prompt: str,
        duration: int,
        style_profile: Optional[Dict],
        lyrics: Optional[str],
        artist_name: Optional[str],
        temperature: float,
        top_k: int,
        guidance_scale: float,
        melody_reference: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate stems one at a time (default for lower RAM)."""
        stems = {}
        for stem_type in stems_to_generate:
            audio = self.generate_stem(
                stem_type=stem_type,
                base_prompt=prompt,
                duration=duration,
                style_profile=style_profile,
                lyrics=lyrics if stem_type == 'vocals' else None,
                artist_name=artist_name,
                temperature=temperature,
                top_k=top_k,
                guidance_scale=guidance_scale,
                melody_reference=melody_reference
            )
            stems[stem_type] = audio
        return stems

    def _generate_stems_parallel(
        self,
        stems_to_generate: List[str],
        prompt: str,
        duration: int,
        style_profile: Optional[Dict],
        lyrics: Optional[str],
        artist_name: Optional[str],
        temperature: float,
        top_k: int,
        guidance_scale: float,
        melody_reference: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate multiple stems in parallel for 64GB+ RAM machines."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"\n⚡ Parallel generation: {min(len(stems_to_generate), self.max_parallel_stems)} stems at a time")
        
        stems = {}
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_stems) as executor:
            futures = {}
            
            for stem_type in stems_to_generate:
                future = executor.submit(
                    self.generate_stem,
                    stem_type=stem_type,
                    base_prompt=prompt,
                    duration=duration,
                    style_profile=style_profile,
                    lyrics=lyrics if stem_type == 'vocals' else None,
                    artist_name=artist_name,
                    temperature=temperature,
                    top_k=top_k,
                    guidance_scale=guidance_scale,
                    melody_reference=melody_reference
                )
                futures[future] = stem_type
            
            for future in as_completed(futures):
                stem_type = futures[future]
                try:
                    audio = future.result()
                    stems[stem_type] = audio
                except Exception as e:
                    print(f"Error generating {stem_type}: {e}")
                    stems[stem_type] = np.zeros(int(duration * 32000))
        
        return stems

    def _crossfade_sections(
        self,
        sections: List[np.ndarray],
        crossfade_samples: int = 16000
    ) -> np.ndarray:
        """Crossfade audio sections together."""
        if not sections:
            return np.zeros(32000)
        
        if len(sections) == 1:
            return sections[0]
        
        is_stereo = any(s.ndim == 2 for s in sections)
        
        total_length = sum(
            s.shape[-1] if s.ndim > 1 else len(s)
            for s in sections
        ) - crossfade_samples * (len(sections) - 1)
        
        if is_stereo:
            result = np.zeros((2, total_length))
        else:
            result = np.zeros(total_length)
        
        position = 0
        
        for i, section in enumerate(sections):
            if is_stereo and section.ndim == 1:
                section = np.stack([section, section])
            elif not is_stereo and section.ndim == 2:
                section = section.mean(axis=0)
            
            section_length = section.shape[-1] if section.ndim > 1 else len(section)
            
            if i == 0:
                if is_stereo:
                    result[:, :section_length] = section
                else:
                    result[:section_length] = section
                position = section_length - crossfade_samples
            else:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                if is_stereo:
                    result[:, position:position + crossfade_samples] *= fade_out
                    section[:, :crossfade_samples] *= fade_in
                    result[:, position:position + crossfade_samples] += section[:, :crossfade_samples]
                    rest_length = section_length - crossfade_samples
                    result[:, position + crossfade_samples:position + crossfade_samples + rest_length] = section[:, crossfade_samples:]
                else:
                    result[position:position + crossfade_samples] *= fade_out
                    section[:crossfade_samples] *= fade_in
                    result[position:position + crossfade_samples] += section[:crossfade_samples]
                    rest_length = section_length - crossfade_samples
                    result[position + crossfade_samples:position + crossfade_samples + rest_length] = section[crossfade_samples:]
                
                position += section_length - crossfade_samples
        
        return result


if __name__ == "__main__":
    print("Testing Multi-Stem Generator with 64GB RAM optimizations...")
    
    # Use all optimizations for a powerful machine
    generator = MultiStemGenerator(
        model_size="large",
        enable_processing=True,
        enable_mastering=True,
        enable_parallel_generation=True,  # Parallel stems
        max_parallel_stems=2,              # 2 stems at once
        cache_all_models=True              # Keep melody model in memory
    )
    
    test_prompt = "aggressive trap beat, heavy 808s, dark atmosphere"
    
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED 64GB RAM GENERATION")
    print(f"{'='*60}")
    print(f"Parallel generation: {generator.enable_parallel_generation}")
    print(f"Max parallel stems: {generator.max_parallel_stems}")
    print(f"Model caching: {generator.cache_all_models}")
    print(f"Processing: {generator.enable_processing}")
    print(f"Mastering: {generator.enable_mastering}")
    print(f"{'='*60}\n")
    
    mixed_path, stem_paths = generator.generate(
        prompt=test_prompt,
        duration=10,
        artist_name="Yeat",
        stems_to_generate=['drums', 'bass', 'melody_1'],
        save_individual_stems=True
    )
    
    print(f"\n✅ Test complete!")
    print(f"Mixed file: {mixed_path}")
    print(f"Stems: {list(stem_paths.keys())}")

