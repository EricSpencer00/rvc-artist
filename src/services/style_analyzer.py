"""
Style Analyzer Service
======================
Analyzes musical features from audio files using librosa.
Extracts tempo, key, energy, spectral features, and creates style profiles.
"""

import numpy as np
import librosa
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class StyleAnalyzer:
    """Analyzes musical style and features from audio files."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio time series
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y
    
    def analyze_tempo(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tempo and beat information.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with tempo information
        """
        # Estimate tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        
        # Get beat times
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
        
        return {
            'tempo': float(tempo),
            'num_beats': len(beats),
            'beat_times': beat_times.tolist() if len(beat_times) < 100 else [],
            'tempo_confidence': 'high' if len(beats) > 10 else 'low'
        }
    
    def analyze_key(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Estimate the musical key using chroma features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with key information
        """
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate)
        
        # Average chroma across time
        chroma_avg = np.mean(chroma, axis=1)
        
        # Find dominant pitch class
        dominant_pitch = np.argmax(chroma_avg)
        
        # Map to note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = note_names[dominant_pitch]
        
        # Simple major/minor estimation based on third
        major_third = (dominant_pitch + 4) % 12
        minor_third = (dominant_pitch + 3) % 12
        
        mode = 'major' if chroma_avg[major_third] > chroma_avg[minor_third] else 'minor'
        
        return {
            'key': estimated_key,
            'mode': mode,
            'full_key': f"{estimated_key} {mode}",
            'chroma_vector': chroma_avg.tolist()
        }
    
    def analyze_energy(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze energy and dynamics.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with energy metrics
        """
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Zero crossing rate (indicates noisiness/percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'dynamic_range': float(np.max(rms) - np.min(rms))
        }
    
    def analyze_spectral(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spectral features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
            'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
            'mfcc_std': np.std(mfccs, axis=1).tolist()
        }
    
    def analyze_rhythm(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze rhythmic features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with rhythm information
        """
        # Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=self.sample_rate)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        
        return {
            'onset_strength_mean': float(np.mean(onset_env)),
            'onset_strength_std': float(np.std(onset_env)),
            'rhythmic_complexity': float(np.std(tempogram))
        }
    
    def analyze_structure(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze song structure.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with structural information
        """
        # Duration
        duration = librosa.get_duration(y=y, sr=self.sample_rate)
        
        # Detect segments using self-similarity
        try:
            # Compute chroma features for structure
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate)
            
            # Compute self-similarity matrix
            R = librosa.segment.recurrence_matrix(chroma, mode='affinity')
            
            # Detect boundaries
            boundaries = librosa.segment.agglomerative(chroma, k=8)
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sample_rate)
            
            num_sections = len(boundary_times)
        except:
            num_sections = 1
            boundary_times = []
        
        return {
            'duration': float(duration),
            'num_sections': int(num_sections),
            'section_boundaries': boundary_times.tolist() if len(boundary_times) < 20 else []
        }
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform complete analysis on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with all features
        """
        print(f"Analyzing: {Path(audio_path).name}")
        
        try:
            # Load audio
            y = self.load_audio(audio_path)
            
            # Perform all analyses
            features = {
                'audio_file': Path(audio_path).name,
                'tempo': self.analyze_tempo(y),
                'key': self.analyze_key(y),
                'energy': self.analyze_energy(y),
                'spectral': self.analyze_spectral(y),
                'rhythm': self.analyze_rhythm(y),
                'structure': self.analyze_structure(y)
            }
            
            print(f"Analysis complete for {Path(audio_path).name}")
            return features
            
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return {
                'audio_file': Path(audio_path).name,
                'error': str(e)
            }
    
    def create_style_profile(self, all_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create an aggregated style profile from multiple song analyses.
        
        Args:
            all_features: List of feature dictionaries from analyze()
            
        Returns:
            Aggregated style profile
        """
        # Filter out any failed analyses
        valid_features = [f for f in all_features if 'error' not in f]
        
        if not valid_features:
            return {'error': 'No valid analyses to aggregate'}
        
        # Aggregate tempo
        tempos = [f['tempo']['tempo'] for f in valid_features]
        
        # Aggregate keys (find most common)
        keys = [f['key']['full_key'] for f in valid_features]
        key_counts = {}
        for key in keys:
            key_counts[key] = key_counts.get(key, 0) + 1
        most_common_key = max(key_counts, key=key_counts.get) if key_counts else 'Unknown'
        
        # Aggregate energy
        rms_means = [f['energy']['rms_mean'] for f in valid_features]
        dynamic_ranges = [f['energy']['dynamic_range'] for f in valid_features]
        
        # Aggregate spectral
        spectral_centroids = [f['spectral']['spectral_centroid_mean'] for f in valid_features]
        
        # Aggregate structure
        durations = [f['structure']['duration'] for f in valid_features]
        
        profile = {
            'num_songs_analyzed': len(valid_features),
            'tempo': {
                'mean': float(np.mean(tempos)),
                'std': float(np.std(tempos)),
                'min': float(np.min(tempos)),
                'max': float(np.max(tempos)),
                'median': float(np.median(tempos))
            },
            'key': {
                'most_common': most_common_key,
                'distribution': key_counts
            },
            'energy': {
                'rms_mean': float(np.mean(rms_means)),
                'rms_std': float(np.std(rms_means)),
                'dynamic_range_mean': float(np.mean(dynamic_ranges))
            },
            'spectral': {
                'brightness_mean': float(np.mean(spectral_centroids)),
                'brightness_std': float(np.std(spectral_centroids))
            },
            'structure': {
                'avg_duration': float(np.mean(durations)),
                'duration_std': float(np.std(durations))
            },
            'characteristics': self._generate_characteristics(
                tempos, rms_means, spectral_centroids
            )
        }
        
        return profile
    
    def _generate_characteristics(
        self,
        tempos: List[float],
        energies: List[float],
        brightnesses: List[float]
    ) -> Dict[str, str]:
        """
        Generate human-readable characteristic descriptions.
        
        Args:
            tempos: List of tempo values
            energies: List of energy values
            brightnesses: List of brightness values
            
        Returns:
            Dictionary of characteristics
        """
        avg_tempo = np.mean(tempos)
        avg_energy = np.mean(energies)
        avg_brightness = np.mean(brightnesses)
        
        # Tempo description
        if avg_tempo < 90:
            tempo_desc = "slow"
        elif avg_tempo < 120:
            tempo_desc = "moderate"
        elif avg_tempo < 140:
            tempo_desc = "upbeat"
        else:
            tempo_desc = "fast"
        
        # Energy description
        if avg_energy < 0.05:
            energy_desc = "calm"
        elif avg_energy < 0.15:
            energy_desc = "moderate energy"
        else:
            energy_desc = "high energy"
        
        # Brightness description
        if avg_brightness < 1500:
            brightness_desc = "warm"
        elif avg_brightness < 2500:
            brightness_desc = "balanced"
        else:
            brightness_desc = "bright"
        
        return {
            'tempo': tempo_desc,
            'energy': energy_desc,
            'timbre': brightness_desc,
            'overall': f"{tempo_desc}, {energy_desc}, {brightness_desc} sound"
        }
