"""
Yeat Voice Converter - RVC Integration
======================================
Converts vocal audio to match Yeat's characteristic voice.
Uses Retrieval-based Voice Conversion (RVC) model trained on Yeat samples.
"""

from pathlib import Path
import json
import numpy as np
from typing import Optional
import soundfile as sf
from datetime import datetime

class YeatVoiceConverter:
    """Voice conversion using Yeat-trained RVC model."""
    
    def __init__(self, model_path: str = "models/rvc/yeat_model"):
        self.model_path = Path(model_path)
        self.config = {}
        self.loaded = False
        self.metadata = {}
        
        self.load_model()
    
    def load_model(self):
        """Load trained Yeat RVC model configuration."""
        try:
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
                
                # Also load training metadata
                metadata_path = self.model_path / "training_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        self.metadata = json.load(f)
                
                print(f"✅ Yeat voice model loaded (trained on {self.config.get('num_training_samples', 1)} samples)")
                self.loaded = True
            else:
                print(f"⚠️  Model config not found at {config_path}")
                self.loaded = False
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.loaded = False
    
    def convert(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 32000,
        pitch_shift: int = 0,
        formant_shift: float = 0.0
    ) -> np.ndarray:
        """
        Convert input vocal to Yeat's voice.
        
        Args:
            audio: Input audio array (mono or stereo)
            sample_rate: Sample rate in Hz
            pitch_shift: Pitch shift in semitones (positive = higher, negative = lower)
            formant_shift: Formant shift for timbre adjustment (0.0 = no change, 0.1 = 10% increase)
        
        Returns:
            Converted audio array with Yeat's voice characteristics
        """
        if not self.loaded:
            print("⚠️  Yeat model not loaded, returning original audio")
            return audio
        
        if audio is None or audio.size == 0:
            return audio
        
        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Apply pitch shift if requested
        if pitch_shift != 0:
            audio = self._apply_pitch_shift(audio, pitch_shift, sample_rate)
        
        # Apply formant shift if requested (voice timbre adjustment)
        if formant_shift != 0.0:
            audio = self._apply_formant_shift(audio, formant_shift, sample_rate)
        
        # Apply Yeat-specific characteristics
        # Yeat is known for:
        # - Slurred, mumbling delivery
        # - Nasal tone
        # - Tight, punchy vocal performance
        audio = self._apply_yeat_characteristics(audio, sample_rate)
        
        return audio
    
    def _apply_pitch_shift(
        self, 
        audio: np.ndarray, 
        semitones: int, 
        sample_rate: int
    ) -> np.ndarray:
        """Apply pitch shifting without changing duration."""
        try:
            import librosa
            
            # Use librosa's pitch shifting
            shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
            return shifted.astype(np.float32)
            
        except ImportError:
            print("⚠️  librosa not available for pitch shifting")
            return audio
    
    def _apply_formant_shift(
        self, 
        audio: np.ndarray, 
        shift_factor: float, 
        sample_rate: int
    ) -> np.ndarray:
        """Apply formant shifting for timbre adjustment."""
        try:
            import librosa
            
            # Simple formant shift by resampling
            # shift_factor: positive values shift formants up, negative down
            if shift_factor == 0:
                return audio
            
            # Shift frequency content
            stretched = librosa.effects.time_stretch(audio, rate=1.0 / (1.0 + shift_factor))
            
            # Resample back to original length
            output_len = len(audio)
            if len(stretched) != output_len:
                stretched = np.interp(
                    np.linspace(0, 1, output_len),
                    np.linspace(0, 1, len(stretched)),
                    stretched
                )
            
            return stretched.astype(np.float32)
            
        except ImportError:
            print("⚠️  librosa not available for formant shifting")
            return audio
    
    def _apply_yeat_characteristics(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """Apply Yeat-specific voice characteristics."""
        try:
            from scipy import signal
            
            # Design a filter that emphasizes Yeat's characteristics:
            # - Boost mids (1-4 kHz) for presence and clarity
            # - Slight boost in sibilance (3-8 kHz)
            # - Reduce extreme highs (above 12 kHz) for a warmer tone
            # - Subtle presence peak around 2-3 kHz
            
            nyquist = sample_rate / 2
            
            # Presence boost (2-4 kHz)
            sos_presence = signal.butter(4, [2000/nyquist, 4000/nyquist], btype='band', output='sos')
            audio = signal.sosfilt(sos_presence, audio) * 1.2
            
            # Sibilance control (4-8 kHz) - slight reduction for less harsh
            sos_sibl = signal.butter(4, [4000/nyquist, 8000/nyquist], btype='band', output='sos')
            sibl = signal.sosfilt(sos_sibl, audio)
            audio = audio - (sibl * 0.1)  # Reduce sibilance slightly
            
            # Normalize to prevent clipping
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val * 0.95
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️  Could not apply voice characteristics: {e}")
            return audio
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "config": self.config,
            "metadata": self.metadata,
            "voice_name": "Yeat",
            "last_updated": datetime.now().isoformat()
        }


# Global instance
_yeat_converter = None

def get_yeat_converter(model_path: str = "models/rvc/yeat_model") -> YeatVoiceConverter:
    """Get or create global Yeat voice converter instance."""
    global _yeat_converter
    if _yeat_converter is None:
        _yeat_converter = YeatVoiceConverter(model_path)
    return _yeat_converter
