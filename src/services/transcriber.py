"""
Audio Transcriber Service
=========================
Transcribes audio files using OpenAI Whisper.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile


class AudioTranscriber:
    """Transcribes audio using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            print(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            print("Whisper model loaded successfully")
        except ImportError:
            print("Whisper not available. Install with: pip install openai-whisper")
            self.model = None
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'es')
            task: 'transcribe' or 'translate'
        
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            return self._fallback_transcribe(audio_path)
        
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path.name}")
        
        try:
            # Transcribe with Whisper
            options = {
                'task': task,
                'verbose': False
            }
            
            if language:
                options['language'] = language
            
            result = self.model.transcribe(str(audio_path), **options)
            
            # Process segments with timestamps
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip(),
                    'confidence': seg.get('no_speech_prob', 0)
                })
            
            return {
                'filename': audio_path.name,
                'language': result.get('language', 'unknown'),
                'text': result['text'].strip(),
                'segments': segments,
                'word_count': len(result['text'].split()),
                'duration': segments[-1]['end'] if segments else 0
            }
            
        except Exception as e:
            print(f"Error transcribing {audio_path.name}: {e}")
            return {
                'filename': audio_path.name,
                'error': str(e),
                'text': '',
                'segments': []
            }
    
    def _fallback_transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Fallback transcription when Whisper is not available."""
        audio_path = Path(audio_path)
        
        return {
            'filename': audio_path.name,
            'language': 'unknown',
            'text': '[Transcription unavailable - Whisper not installed]',
            'segments': [],
            'word_count': 0,
            'duration': 0,
            'note': 'Install openai-whisper for actual transcription'
        }
    
    def transcribe_batch(
        self,
        audio_files: List[str],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Optional directory to save transcriptions
        
        Returns:
            List of transcription results
        """
        results = []
        
        for i, audio_path in enumerate(audio_files):
            print(f"Transcribing {i + 1}/{len(audio_files)}: {Path(audio_path).name}")
            
            result = self.transcribe(audio_path)
            results.append(result)
            
            # Save individual transcript if output_dir specified
            if output_dir:
                output_path = Path(output_dir) / f"{Path(audio_path).stem}.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return results
    
    def get_word_timestamps(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Get word-level timestamps for an audio file.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            List of words with timestamps
        """
        if self.model is None:
            return []
        
        try:
            import whisper
            
            # Load audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            # Decode with word timestamps
            options = whisper.DecodingOptions(
                language=detected_lang,
                without_timestamps=False
            )
            
            result = whisper.decode(self.model, mel, options)
            
            # Note: Word-level timestamps require additional processing
            # This is a simplified implementation
            words = []
            for token in result.tokens:
                if hasattr(token, 'word') and hasattr(token, 'start'):
                    words.append({
                        'word': token.word,
                        'start': token.start,
                        'end': token.end
                    })
            
            return words
            
        except Exception as e:
            print(f"Error getting word timestamps: {e}")
            return []


class VocalSeparator:
    """Separates vocals from instrumentals using Demucs or similar."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the vocal separation model."""
        try:
            # Try to use demucs for vocal separation
            import demucs.separate
            self.model = "demucs"
            print("Demucs available for vocal separation")
        except ImportError:
            print("Demucs not available. Vocal separation will be skipped.")
            self.model = None
    
    def separate(
        self,
        audio_path: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Separate vocals from instrumentals.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save separated audio
        
        Returns:
            Dictionary with paths to vocals and instrumentals
        """
        if self.model is None:
            return {
                'vocals': audio_path,
                'instrumentals': None,
                'note': 'Vocal separation not available'
            }
        
        try:
            import subprocess
            
            audio_path = Path(audio_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run demucs
            result = subprocess.run(
                [
                    'python', '-m', 'demucs.separate',
                    '-n', 'htdemucs',
                    '-o', str(output_dir),
                    str(audio_path)
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Find output files
                stem_name = audio_path.stem
                separated_dir = output_dir / 'htdemucs' / stem_name
                
                return {
                    'vocals': str(separated_dir / 'vocals.wav'),
                    'instrumentals': str(separated_dir / 'no_vocals.wav'),
                    'bass': str(separated_dir / 'bass.wav'),
                    'drums': str(separated_dir / 'drums.wav'),
                    'other': str(separated_dir / 'other.wav')
                }
            else:
                print(f"Demucs error: {result.stderr}")
                return {'vocals': audio_path, 'instrumentals': None}
                
        except Exception as e:
            print(f"Error separating vocals: {e}")
            return {'vocals': audio_path, 'instrumentals': None}


if __name__ == "__main__":
    # Test the transcriber
    transcriber = AudioTranscriber(model_size="base")
    
    # Test with a sample file if available
    test_dir = Path("./data/audio/raw")
    if test_dir.exists():
        mp3_files = list(test_dir.glob("*.mp3"))
        if mp3_files:
            print(f"Testing with: {mp3_files[0].name}")
            result = transcriber.transcribe(str(mp3_files[0]))
            print(f"Transcription preview: {result['text'][:200]}...")
