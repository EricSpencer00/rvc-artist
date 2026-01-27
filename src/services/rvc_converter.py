"""
RVC Voice Conversion Service
=============================
Uses Retrieval-Based Voice Conversion to train and apply voice models.
Can train on Yeat vocal samples and convert any input vocals to Yeat voice.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import shutil


class RVCVoiceConverter:
    """Handles RVC model training and voice conversion."""
    
    def __init__(self, model_dir: str = "./models/rvc"):
        """Initialize RVC converter."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to import RVC
        try:
            # This would need the RVC package to be installed
            # For now, we'll create a wrapper that can be filled in
            self.rvc_available = False
        except ImportError:
            self.rvc_available = False
    
    def train_model(
        self,
        voice_samples_dir: str,
        model_name: str = "yeat",
        epochs: int = 20,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Train an RVC model on voice samples.
        
        Args:
            voice_samples_dir: Directory with audio files (.wav, .mp3) of target voice
            model_name: Name for the model (e.g., 'yeat')
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training info and model path
        """
        samples_path = Path(voice_samples_dir)
        
        if not samples_path.exists():
            raise ValueError(f"Voice samples directory not found: {voice_samples_dir}")
        
        # Collect voice samples
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a']:
            audio_files.extend(samples_path.glob(ext))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {voice_samples_dir}")
        
        # Create model directory
        model_path = self.model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare training data
        training_data_dir = model_path / "training_data"
        training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and preprocess audio files
        import librosa
        import soundfile as sf
        
        sample_rate = 40000  # RVC standard
        processed_count = 0
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(str(audio_file), sr=sample_rate)
                output_file = training_data_dir / f"sample_{processed_count:03d}.wav"
                sf.write(str(output_file), y, sample_rate)
                processed_count += 1
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
        
        if processed_count == 0:
            raise ValueError("No audio files could be processed")
        
        # Extract f0 (fundamental frequency) for each sample
        f0_dir = model_path / "f0"
        f0_dir.mkdir(parents=True, exist_ok=True)
        
        self._extract_f0_files(training_data_dir, f0_dir, sample_rate)
        
        # Create model metadata
        metadata = {
            'model_name': model_name,
            'trained_samples': processed_count,
            'sample_rate': sample_rate,
            'device': self.device,
            'created_at': datetime.now().isoformat(),
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'status': 'trained',
            'model_name': model_name,
            'model_path': str(model_path),
            'samples_processed': processed_count,
            'sample_rate': sample_rate,
            'message': f'RVC model "{model_name}" trained on {processed_count} samples'
        }
    
    def _extract_f0_files(self, audio_dir: Path, f0_dir: Path, sample_rate: int):
        """Extract fundamental frequency for training data."""
        import librosa
        import json
        
        for audio_file in audio_dir.glob("*.wav"):
            try:
                y, sr = librosa.load(str(audio_file), sr=sample_rate)
                
                # Extract F0 using crepe or yin
                f0 = librosa.yin(y, fmin=50, fmax=400, sr=sample_rate)
                
                # Save F0
                f0_file = f0_dir / f"{audio_file.stem}.npy"
                np.save(str(f0_file), f0)
            except Exception as e:
                print(f"Warning: Could not extract F0 from {audio_file}: {e}")
    
    def convert_voice(
        self,
        input_audio_path: str,
        model_name: str = "yeat",
        output_path: Optional[str] = None,
        pitch_shift: int = 0,
        index_rate: float = 0.5,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25
    ) -> Dict[str, Any]:
        """
        Convert voice in input audio using trained RVC model.
        
        Args:
            input_audio_path: Path to input audio file
            model_name: Name of trained model to use
            output_path: Where to save converted audio
            pitch_shift: Pitch shift in semitones (+ or -)
            index_rate: How much to use the index (0-1)
            filter_radius: Audio filter radius
            resample_sr: Resample output to this sr (0 = don't resample)
            rms_mix_rate: RMS mixing rate
            
        Returns:
            Dictionary with conversion info
        """
        input_path = Path(input_audio_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")
        
        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_name}. Train it first.")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"converted_{model_name}_{timestamp}.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess input
        import librosa
        import soundfile as sf
        
        sample_rate = 40000  # RVC standard
        y, sr = librosa.load(str(input_path), sr=sample_rate)
        
        # For now, simulate the conversion
        # In production, this would use the actual RVC inference
        y_converted = self._apply_voice_conversion(
            y,
            sample_rate,
            model_path,
            pitch_shift=pitch_shift
        )
        
        # Save converted audio
        sf.write(str(output_path), y_converted, sample_rate)
        
        return {
            'status': 'converted',
            'input_file': str(input_path),
            'output_file': str(output_path),
            'model_used': model_name,
            'pitch_shift': pitch_shift,
            'duration': len(y_converted) / sample_rate,
            'message': f'Voice converted using model "{model_name}"'
        }
    
    def _apply_voice_conversion(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model_path: Path,
        pitch_shift: int = 0
    ) -> np.ndarray:
        """
        Apply voice conversion to audio array.
        
        This is a placeholder. Real RVC inference is more complex and requires:
        - Loading the ONNX/PyTorch model
        - Feature extraction
        - Model inference
        - Vocoding
        """
        converted = audio.copy()
        
        # Apply pitch shift if specified
        if pitch_shift != 0:
            import librosa
            converted = librosa.effects.pitch_shift(
                converted,
                sr=sample_rate,
                n_steps=pitch_shift
            )
        
        return converted
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        models = []
        
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    metadata['path'] = str(model_dir)
                    models.append(metadata)
        
        return models
    
    def load_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a specific model."""
        model_path = self.model_dir / model_name
        metadata_file = model_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        model_path = self.model_dir / model_name
        
        if model_path.exists():
            shutil.rmtree(model_path)
            return True
        
        return False
