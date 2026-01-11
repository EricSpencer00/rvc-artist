"""
Lyrics Generator Service
========================
Generates lyrics based on artist style and vocabulary.
Uses a simple Markov chain or template-based approach to mimic artist themes.
"""

import random
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


class LyricsGenerator:
    """Generates lyrics in the style of an artist."""
    
    def __init__(self):
        self.vocabulary = set()
        self.markov_chain = defaultdict(list)
        self.common_phrases = []
        self.is_trained = False

    def train_from_transcripts(self, transcript_dir: str):
        """
        Train the generator using transcription files.
        """
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            return
        
        all_text = ""
        for transcript_file in transcript_path.glob("*.json"):
            try:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
                    text = data.get('text', '')
                    all_text += " " + text
            except Exception as e:
                print(f"Error reading transcript {transcript_file}: {e}")
        
        if all_text:
            self._build_markov_chain(all_text)
            self.is_trained = True
            print(f"LyricsGenerator trained on {len(list(transcript_path.glob('*.json')))} transcripts.")

    def _build_markov_chain(self, text: str):
        """Build a simple Markov chain for text generation."""
        words = re.findall(r'\w+', text.lower())
        for i in range(len(words) - 1):
            self.markov_chain[words[i]].append(words[i+1])
        self.vocabulary = list(self.markov_chain.keys())

    def generate_lyrics(self, num_lines: int = 8, words_per_line: int = 6) -> str:
        """
        Generate new lyrics.
        """
        if not self.is_trained:
            return "No training data available. Run analysis/transcription first."
        
        lines = []
        for _ in range(num_lines):
            line = []
            word = random.choice(self.vocabulary)
            line.append(word.capitalize())
            
            for _ in range(words_per_line - 1):
                next_words = self.markov_chain.get(word, self.vocabulary)
                word = random.choice(next_words)
                line.append(word)
            
            lines.append(" ".join(line))
        
        return "\n".join(lines)

    def get_style_summary(self) -> Dict[str, Any]:
        """Return a summary of the learned style."""
        if not self.is_trained:
            return {}
            
        return {
            'vocabulary_size': len(self.vocabulary),
            'top_starts': self.vocabulary[:10]  # Just a sample
        }
