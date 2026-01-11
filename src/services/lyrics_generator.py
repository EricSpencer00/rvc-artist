"""
Lyrics Generator Service
========================
Generates lyrics based on artist style and vocabulary.
Uses n-gram Markov chains with rhyme awareness and keyword conditioning.
"""

import random
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter


class LyricsGenerator:
    """Generates lyrics in the style of an artist with rhyme and structure awareness."""
    
    # Common rhyme endings for grouping
    RHYME_ENDINGS = {
        'ay': ['day', 'way', 'say', 'play', 'stay', 'pay', 'today', 'away', 'okay', 'yay', 'hey', 'they'],
        'ight': ['night', 'right', 'light', 'fight', 'tight', 'sight', 'bright', 'flight', 'tonight'],
        'ow': ['know', 'go', 'flow', 'show', 'low', 'blow', 'grow', 'throw', 'glow', 'slow'],
        'ee': ['me', 'be', 'see', 'free', 'we', 'she', 'three', 'key', 'tree', 'flee'],
        'ine': ['mine', 'line', 'time', 'fine', 'shine', 'wine', 'sign', 'divine', 'rhyme', 'dime'],
        'ame': ['game', 'name', 'fame', 'same', 'flame', 'blame', 'came', 'shame', 'claim'],
        'ound': ['sound', 'round', 'ground', 'found', 'pound', 'bound', 'around', 'down', 'town', 'crown'],
        'ack': ['back', 'track', 'stack', 'black', 'crack', 'pack', 'lack', 'attack', 'whack'],
        'op': ['top', 'drop', 'stop', 'hop', 'pop', 'shop', 'chop', 'crop', 'nonstop'],
        'eal': ['real', 'feel', 'deal', 'steal', 'wheel', 'heal', 'reveal', 'seal'],
        'all': ['all', 'ball', 'call', 'fall', 'wall', 'tall', 'small', 'hall', 'mall'],
        'ove': ['love', 'above', 'dove', 'shove', 'glove', 'of'],
        'ain': ['rain', 'pain', 'brain', 'chain', 'main', 'train', 'gain', 'insane', 'maintain'],
        'ess': ['less', 'stress', 'bless', 'press', 'mess', 'dress', 'success', 'guess', 'impress'],
        'ap': ['trap', 'cap', 'rap', 'map', 'snap', 'clap', 'gap', 'wrap', 'slap'],
    }
    
    # Section types for structured generation
    SECTION_TYPES = ['verse', 'chorus', 'hook', 'bridge', 'intro', 'outro']
    
    def __init__(self, ngram_size: int = 3):
        """
        Initialize the generator.
        
        Args:
            ngram_size: Size of n-grams for Markov chain (2=bigram, 3=trigram)
        """
        self.ngram_size = ngram_size
        self.vocabulary = set()
        self.markov_chain = defaultdict(list)  # For bigrams
        self.trigram_chain = defaultdict(list)  # For trigrams
        self.word_freq = Counter()
        self.rhyme_groups = defaultdict(set)
        self.line_starters = []
        self.common_phrases = []
        self.style_keywords = []
        self.is_trained = False

    def train_from_transcripts(self, transcript_dir: str):
        """
        Train the generator using transcription files.
        
        Args:
            transcript_dir: Directory containing transcript JSON files
        """
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            print(f"⚠️ Transcript directory not found: {transcript_dir}")
            return
        
        all_text = ""
        all_lines = []
        
        for transcript_file in transcript_path.glob("*.json"):
            try:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
                    text = data.get('text', '')
                    all_text += " " + text
                    
                    # Try to get individual segments for line awareness
                    segments = data.get('segments', [])
                    for seg in segments:
                        seg_text = seg.get('text', '').strip()
                        if seg_text and len(seg_text.split()) >= 3:
                            all_lines.append(seg_text)
                    
                    # Also split by common patterns
                    lines = re.split(r'[.!?\n]+', text)
                    all_lines.extend([l.strip() for l in lines if l.strip() and len(l.split()) >= 3])
                    
            except Exception as e:
                print(f"Error reading transcript {transcript_file}: {e}")
        
        if all_text:
            self._build_chains(all_text)
            self._extract_line_starters(all_lines)
            self._build_rhyme_groups()
            self.is_trained = True
            num_files = len(list(transcript_path.glob('*.json')))
            print(f"✅ LyricsGenerator trained on {num_files} transcripts")
            print(f"   Vocabulary: {len(self.vocabulary)} words")
            print(f"   Rhyme groups: {len(self.rhyme_groups)} patterns")

    def _build_chains(self, text: str):
        """Build bigram and trigram Markov chains."""
        words = re.findall(r'\w+', text.lower())
        self.word_freq = Counter(words)
        self.vocabulary = set(words)
        
        # Build bigram chain
        for i in range(len(words) - 1):
            self.markov_chain[words[i]].append(words[i+1])
        
        # Build trigram chain
        for i in range(len(words) - 2):
            key = (words[i], words[i+1])
            self.trigram_chain[key].append(words[i+2])

    def _extract_line_starters(self, lines: List[str]):
        """Extract common line starting patterns."""
        starters = []
        for line in lines:
            words = line.split()
            if len(words) >= 2:
                starters.append((words[0].lower(), words[1].lower()))
            if len(words) >= 1:
                starters.append((words[0].lower(),))
        
        # Keep most common starters
        starter_counts = Counter(starters)
        self.line_starters = [s for s, _ in starter_counts.most_common(50)]

    def _build_rhyme_groups(self):
        """Group vocabulary words by rhyme endings."""
        for word in self.vocabulary:
            if len(word) < 2:
                continue
            
            # Check against known rhyme patterns
            for ending, _ in self.RHYME_ENDINGS.items():
                if word.endswith(ending) or word[-2:] == ending[-2:] if len(ending) >= 2 else False:
                    self.rhyme_groups[ending].add(word)
                    break
            else:
                # Use last 2-3 characters as rhyme key
                if len(word) >= 3:
                    self.rhyme_groups[word[-3:]].add(word)
                elif len(word) >= 2:
                    self.rhyme_groups[word[-2:]].add(word)

    def _get_rhyming_word(self, target_word: str) -> Optional[str]:
        """Find a word that rhymes with the target."""
        target = target_word.lower()
        
        # Find which rhyme group the target belongs to
        for ending, words in self.rhyme_groups.items():
            if target in words or target.endswith(ending):
                candidates = list(words - {target})
                if candidates:
                    # Prefer more common words
                    weighted = [(w, self.word_freq.get(w, 1)) for w in candidates]
                    weighted.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [w for w, _ in weighted[:10]]
                    return random.choice(top_candidates) if top_candidates else None
        
        return None

    def _generate_line_trigram(self, words_per_line: int = 6, keyword_bias: Optional[List[str]] = None) -> str:
        """Generate a line using trigram chain."""
        line = []
        
        # Start with a line starter or random start
        if self.line_starters and random.random() < 0.7:
            starter = random.choice(self.line_starters)
            line.extend(starter)
        else:
            # Random start
            word = random.choice(list(self.vocabulary))
            line.append(word)
        
        # Continue with trigram chain
        attempts = 0
        while len(line) < words_per_line and attempts < 20:
            attempts += 1
            
            # Try trigram first
            if len(line) >= 2:
                key = (line[-2], line[-1])
                if key in self.trigram_chain:
                    candidates = self.trigram_chain[key]
                    
                    # Apply keyword bias
                    if keyword_bias:
                        biased = [w for w in candidates if w in keyword_bias]
                        if biased and random.random() < 0.3:
                            line.append(random.choice(biased))
                            continue
                    
                    line.append(random.choice(candidates))
                    continue
            
            # Fall back to bigram
            if line:
                key = line[-1]
                if key in self.markov_chain:
                    candidates = self.markov_chain[key]
                    
                    # Apply keyword bias
                    if keyword_bias:
                        biased = [w for w in candidates if w in keyword_bias]
                        if biased and random.random() < 0.3:
                            line.append(random.choice(biased))
                            continue
                    
                    line.append(random.choice(candidates))
                    continue
            
            # Last resort: random word
            if keyword_bias and random.random() < 0.4:
                line.append(random.choice(keyword_bias))
            else:
                line.append(random.choice(list(self.vocabulary)))
        
        # Capitalize first word
        if line:
            line[0] = line[0].capitalize()
        
        return " ".join(line)

    def generate_lyrics(
        self,
        num_lines: int = 8,
        words_per_line: int = 6,
        rhyme_scheme: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> str:
        """
        Generate new lyrics with optional rhyme scheme.
        
        Args:
            num_lines: Number of lines to generate
            words_per_line: Average words per line
            rhyme_scheme: Optional rhyme pattern like 'AABB', 'ABAB', 'ABBA'
            keywords: Optional list of keywords to bias generation toward
            
        Returns:
            Generated lyrics as a string
        """
        if not self.is_trained:
            return "No training data available. Run analysis/transcription first."
        
        # Default rhyme scheme for 8 lines
        if rhyme_scheme is None:
            if num_lines >= 4:
                rhyme_scheme = 'AABB' * (num_lines // 4)
            else:
                rhyme_scheme = 'A' * num_lines
        
        # Pad or trim rhyme scheme
        rhyme_scheme = (rhyme_scheme * ((num_lines // len(rhyme_scheme)) + 1))[:num_lines]
        
        lines = []
        rhyme_targets = {}  # Maps rhyme letter to target word
        
        for i in range(num_lines):
            rhyme_letter = rhyme_scheme[i] if i < len(rhyme_scheme) else 'X'
            
            # Generate base line
            line = self._generate_line_trigram(words_per_line, keyword_bias=keywords)
            
            # Try to enforce rhyme if we have a target
            if rhyme_letter in rhyme_targets:
                target_word = rhyme_targets[rhyme_letter]
                rhyming_word = self._get_rhyming_word(target_word)
                
                if rhyming_word:
                    # Replace last word with rhyming word
                    words = line.split()
                    if words:
                        words[-1] = rhyming_word
                        line = " ".join(words)
            else:
                # Store this line's last word as the rhyme target
                words = line.split()
                if words:
                    rhyme_targets[rhyme_letter] = words[-1].lower()
            
            lines.append(line)
        
        return "\n".join(lines)

    def generate_section(
        self,
        section_type: str = 'verse',
        style_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate lyrics for a specific song section.
        
        Args:
            section_type: Type of section ('verse', 'chorus', 'hook', 'bridge')
            style_profile: Optional style profile for keyword conditioning
            
        Returns:
            Generated section lyrics
        """
        # Section-specific parameters
        section_params = {
            'verse': {'lines': 8, 'words': 7, 'scheme': 'AABBCCDD'},
            'chorus': {'lines': 4, 'words': 5, 'scheme': 'AABB'},
            'hook': {'lines': 2, 'words': 4, 'scheme': 'AA'},
            'bridge': {'lines': 4, 'words': 6, 'scheme': 'ABAB'},
            'intro': {'lines': 2, 'words': 4, 'scheme': 'AA'},
            'outro': {'lines': 2, 'words': 4, 'scheme': 'AA'},
        }
        
        params = section_params.get(section_type, section_params['verse'])
        
        # Extract keywords from style profile
        keywords = None
        if style_profile:
            lyrics_info = style_profile.get('lyrics', {})
            keywords = lyrics_info.get('top_keywords', [])
        
        return self.generate_lyrics(
            num_lines=params['lines'],
            words_per_line=params['words'],
            rhyme_scheme=params['scheme'],
            keywords=keywords
        )

    def generate_full_song(
        self,
        structure: Optional[List[str]] = None,
        style_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate lyrics for a full song with structure.
        
        Args:
            structure: List of section types, e.g., ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro']
            style_profile: Optional style profile for conditioning
            
        Returns:
            Dictionary with structure and lyrics
        """
        if structure is None:
            structure = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro']
        
        song = {
            'structure': structure,
            'sections': [],
            'full_lyrics': ''
        }
        
        all_lyrics = []
        
        for section_type in structure:
            lyrics = self.generate_section(section_type, style_profile)
            song['sections'].append({
                'type': section_type,
                'lyrics': lyrics
            })
            all_lyrics.append(f"[{section_type.upper()}]")
            all_lyrics.append(lyrics)
            all_lyrics.append('')
        
        song['full_lyrics'] = '\n'.join(all_lyrics)
        
        return song

    def get_style_summary(self) -> Dict[str, Any]:
        """Return a summary of the learned style."""
        if not self.is_trained:
            return {}
        
        # Get most common words (excluding very common ones)
        stop_words = {'i', 'the', 'a', 'to', 'and', 'is', 'it', 'you', 'that', 'of', 'in', 'my', 'for', 'on'}
        filtered_freq = {w: c for w, c in self.word_freq.items() if w not in stop_words and len(w) > 2}
        top_words = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'vocabulary_size': len(self.vocabulary),
            'unique_bigrams': len(self.markov_chain),
            'unique_trigrams': len(self.trigram_chain),
            'rhyme_patterns': len(self.rhyme_groups),
            'line_starters': len(self.line_starters),
            'top_words': [w for w, _ in top_words],
            'sample_starters': self.line_starters[:5] if self.line_starters else []
        }
