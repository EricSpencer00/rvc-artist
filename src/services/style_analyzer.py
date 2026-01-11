"""
Style Analyzer Service
======================
Analyzes musical features from audio files using librosa (if available).
Extracts tempo, key, energy, spectral features, and creates style profiles.
Supports named profiles for different song subsets.
Falls back to basic analysis if librosa is not installed.
"""

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')


class StyleAnalyzer:
    """Analyzes musical style and features from audio files."""
    
    # Common stop words to filter out from keywords
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
        'yeah', 'got', 'know', 'like', 'get', 'wanna', 'gonna', 'go', 'make', 'take', 'back'
    }

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        if not HAS_LIBROSA:
            print("⚠️  librosa not installed. Install scipy and librosa for full audio analysis.")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio time series
        """
        if not HAS_LIBROSA:
            raise RuntimeError("librosa not available")
        
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return y
    
    def analyze_tempo(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tempo and beats.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with tempo information
        """
        # Estimate tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
        
        # Tempo confidence estimation (simplified)
        if len(beats) > 10:
            confidence = 'high'
        elif len(beats) > 5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'tempo': float(tempo),
            'num_beats': int(len(beats)),
            'beat_times': beat_times.tolist()[:50] if len(beat_times) < 50 else [],  # Limit size
            'tempo_confidence': confidence
        }
    
    def analyze_key(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze musical key using chroma features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary with key information
        """
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        
        # Map chroma to note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        dominant_pitch = int(np.argmax(chroma_avg))
        estimated_key = note_names[dominant_pitch]
        
        # Estimate major vs minor (simplified heuristic)
        major_third = (dominant_pitch + 4) % 12
        minor_third = (dominant_pitch + 3) % 12
        
        mode = 'major' if chroma_avg[major_third] > chroma_avg[minor_third] else 'minor'
        
        return {
            'key': estimated_key,
            'mode': mode,
            'full_key': f"{estimated_key} {mode}",
            'chroma_vector': chroma_avg.tolist()
        }
    
    def analyze(self, audio_path: str, transcript_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform complete analysis on an audio file.
        
        Args:
            audio_path: Path to audio file
            transcript_path: Optional path to transcript JSON
            
        Returns:
            Dictionary with all features
        """
        print(f"Analyzing: {Path(audio_path).name}")
        
        if not HAS_LIBROSA:
            print("⚠️  librosa not installed, returning mock data")
            return {
                'audio_file': Path(audio_path).name,
                'tempo': {'tempo': 120.0, 'num_beats': 32, 'beat_times': [], 'tempo_confidence': 'medium'},
                'key': {'key': 'C', 'mode': 'major', 'full_key': 'C major', 'chroma_vector': [0.1] * 12},
                'energy': {'rms_mean': 0.1, 'rms_std': 0.05, 'rms_max': 0.2, 'zcr_mean': 0.1, 'zcr_std': 0.05, 'dynamic_range': 0.15},
                'spectral': {'spectral_centroid_mean': 2000.0, 'spectral_centroid_std': 500.0, 'spectral_rolloff_mean': 4000.0, 'spectral_bandwidth_mean': 1000.0, 'spectral_contrast_mean': [1.0] * 7, 'mfcc_mean': [0.0] * 13, 'mfcc_std': [0.1] * 13},
                'rhythm': {'onset_strength_mean': 0.1, 'onset_strength_std': 0.05, 'rhythmic_complexity': 0.1},
                'structure': {'duration': 180.0, 'num_sections': 4, 'section_boundaries': []}
            }
        
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

            # Optional transcript analysis
            if transcript_path and Path(transcript_path).exists():
                features['lyrics'] = self.extract_lyrics_keywords(transcript_path)
            
            print(f"✅ Analysis complete for {Path(audio_path).name}")
            return features
            
        except Exception as e:
            print(f"❌ Error analyzing {audio_path}: {e}")
            return {
                'audio_file': Path(audio_path).name,
                'error': str(e)
            }

    def extract_lyrics_keywords(self, transcript_path: str) -> Dict[str, Any]:
        """
        Extract keywords and themes from a transcript.
        """
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
            
            text = data.get('text', '')
            if not text:
                return {'keywords': [], 'themes': []}
            
            # Simple keyword extraction
            words = re.findall(r'\w+', text.lower())
            filtered_words = [w for w in words if len(w) > 3 and w not in self.STOP_WORDS]
            
            common_words = Counter(filtered_words).most_common(10)
            keywords = [word for word, count in common_words]
            
            return {
                'keywords': keywords,
                'word_count': len(words)
            }
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return {'keywords': [], 'error': str(e)}

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

        # Aggregate lyrical keywords
        all_keywords = []
        for f in valid_features:
            if 'lyrics' in f and 'keywords' in f['lyrics']:
                all_keywords.extend(f['lyrics']['keywords'])
        
        top_keywords = [word for word, count in Counter(all_keywords).most_common(12)]
        
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
            'lyrics': {
                'top_keywords': top_keywords
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

    def features_to_prompt_descriptors(self, profile: Dict[str, Any]) -> List[str]:
        """
        Convert a style profile into rich, descriptive prompt phrases.
        
        Args:
            profile: Style profile dictionary
            
        Returns:
            List of descriptive phrases for prompt construction
        """
        descriptors = []
        
        # Tempo descriptors
        tempo = profile.get('tempo', {})
        tempo_mean = tempo.get('mean', 120)
        tempo_std = tempo.get('std', 10)
        
        if tempo_mean < 80:
            descriptors.append("slow groove")
        elif tempo_mean < 100:
            descriptors.append("mid-tempo laid-back feel")
        elif tempo_mean < 120:
            descriptors.append("steady driving rhythm")
        elif tempo_mean < 140:
            descriptors.append("upbeat energetic tempo")
        elif tempo_mean < 160:
            descriptors.append("fast aggressive tempo")
        else:
            descriptors.append("hyper-speed tempo")
        
        # Tempo consistency
        if tempo_std < 5:
            descriptors.append("consistent locked groove")
        elif tempo_std > 15:
            descriptors.append("varied tempo changes")
        
        # Energy descriptors
        energy = profile.get('energy', {})
        rms_mean = energy.get('rms_mean', 0.1)
        dynamic_range = energy.get('dynamic_range_mean', 0.1)
        
        if rms_mean < 0.05:
            descriptors.append("soft intimate production")
        elif rms_mean < 0.1:
            descriptors.append("balanced dynamics")
        elif rms_mean < 0.15:
            descriptors.append("loud punchy mix")
        else:
            descriptors.append("heavily compressed loud master")
        
        if dynamic_range > 0.15:
            descriptors.append("huge dynamic range with punchy drops")
        elif dynamic_range < 0.05:
            descriptors.append("consistent sustained energy")
        
        # Spectral/brightness descriptors
        spectral = profile.get('spectral', {})
        brightness = spectral.get('brightness_mean', 2000)
        
        if brightness < 1200:
            descriptors.append("warm dark low-end heavy")
        elif brightness < 1800:
            descriptors.append("warm balanced frequencies")
        elif brightness < 2500:
            descriptors.append("clear balanced mix")
        elif brightness < 3500:
            descriptors.append("bright crisp highs")
        else:
            descriptors.append("sharp piercing treble")
        
        # Structure descriptors
        structure = profile.get('structure', {})
        avg_duration = structure.get('avg_duration', 180)
        
        if avg_duration < 120:
            descriptors.append("short punchy arrangement")
        elif avg_duration < 180:
            descriptors.append("standard song structure")
        elif avg_duration < 240:
            descriptors.append("extended arrangement")
        else:
            descriptors.append("long evolving composition")
        
        # Lyrical theme descriptors
        lyrics = profile.get('lyrics', {})
        keywords = lyrics.get('top_keywords', [])
        if keywords:
            # Group common themes
            money_words = {'money', 'cash', 'rich', 'bands', 'racks', 'millions', 'dollars'}
            flex_words = {'drip', 'ice', 'chains', 'designer', 'gucci', 'benz', 'lambo'}
            dark_words = {'dark', 'pain', 'demons', 'death', 'blood', 'hate', 'evil'}
            love_words = {'love', 'heart', 'baby', 'girl', 'miss', 'feel'}
            hype_words = {'yeah', 'lit', 'turnt', 'gang', 'squad', 'rage'}
            
            keyword_set = set(k.lower() for k in keywords)
            
            if keyword_set & money_words:
                descriptors.append("wealth flex themes")
            if keyword_set & flex_words:
                descriptors.append("luxury lifestyle imagery")
            if keyword_set & dark_words:
                descriptors.append("dark aggressive mood")
            if keyword_set & love_words:
                descriptors.append("emotional melodic themes")
            if keyword_set & hype_words:
                descriptors.append("hype party energy")
        
        return descriptors

    def save_named_profile(
        self,
        profile: Dict[str, Any],
        profile_name: str,
        output_dir: str = "./data/features"
    ) -> str:
        """
        Save a style profile with a specific name.
        
        Args:
            profile: Style profile dictionary
            profile_name: Name identifier for this profile
            output_dir: Directory to save the profile
            
        Returns:
            Path to the saved profile
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize profile name
        safe_name = re.sub(r'[^\w\-_]', '_', profile_name.lower())
        filename = f"style_profile_{safe_name}.json"
        filepath = output_path / filename
        
        # Add metadata
        profile['profile_name'] = profile_name
        profile['created_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print(f"✅ Saved named profile: {filepath}")
        return str(filepath)

    def load_named_profile(
        self,
        profile_name: str,
        profiles_dir: str = "./data/features"
    ) -> Optional[Dict[str, Any]]:
        """
        Load a named style profile.
        
        Args:
            profile_name: Name identifier for the profile
            profiles_dir: Directory containing profiles
            
        Returns:
            Style profile dictionary or None if not found
        """
        profiles_path = Path(profiles_dir)
        
        # Try exact match first
        safe_name = re.sub(r'[^\w\-_]', '_', profile_name.lower())
        filepath = profiles_path / f"style_profile_{safe_name}.json"
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Try default profile
        default_path = profiles_path / "style_profile.json"
        if default_path.exists():
            with open(default_path, 'r') as f:
                return json.load(f)
        
        return None

    def list_profiles(self, profiles_dir: str = "./data/features") -> List[Dict[str, str]]:
        """
        List all available style profiles.
        
        Args:
            profiles_dir: Directory containing profiles
            
        Returns:
            List of profile info dictionaries
        """
        profiles_path = Path(profiles_dir)
        profiles = []
        
        if not profiles_path.exists():
            return profiles
        
        for filepath in profiles_path.glob("style_profile*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                profiles.append({
                    'filename': filepath.name,
                    'profile_name': data.get('profile_name', 'default'),
                    'num_songs': data.get('num_songs_analyzed', 0),
                    'created_at': data.get('created_at', 'unknown'),
                    'path': str(filepath)
                })
            except Exception:
                pass
        
        return profiles

    def analyze_subset(
        self,
        audio_files: List[str],
        profile_name: str,
        transcripts_dir: Optional[str] = None,
        output_dir: str = "./data/features"
    ) -> Dict[str, Any]:
        """
        Analyze a specific subset of audio files and create a named profile.
        
        Args:
            audio_files: List of audio file paths to analyze
            profile_name: Name for this profile
            transcripts_dir: Optional directory with transcripts
            output_dir: Directory to save profile
            
        Returns:
            The created style profile
        """
        all_features = []
        
        for audio_path in audio_files:
            audio_file = Path(audio_path)
            
            # Look for matching transcript
            transcript_path = None
            if transcripts_dir:
                t_path = Path(transcripts_dir) / f"{audio_file.stem}.json"
                if t_path.exists():
                    transcript_path = str(t_path)
            
            features = self.analyze(str(audio_file), transcript_path=transcript_path)
            all_features.append(features)
        
        # Create and save the profile
        profile = self.create_style_profile(all_features)
        profile['source_files'] = [Path(f).name for f in audio_files]
        
        self.save_named_profile(profile, profile_name, output_dir)
        
        return profile
