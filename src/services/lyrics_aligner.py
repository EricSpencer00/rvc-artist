"""
Lyrics Aligner Service
======================
Aligns transcribed audio with official lyrics using fuzzy matching.
This helps correct transcription errors and map time codes to lyric lines.
"""

import difflib
from typing import Dict, Any, List, Tuple, Optional
from fuzzywuzzy import fuzz
import re


class LyricsAligner:
    """Aligns transcriptions with official lyrics."""
    
    def __init__(self, similarity_threshold: int = 60):
        """
        Initialize the aligner.
        
        Args:
            similarity_threshold: Minimum similarity score (0-100) to consider a match
        """
        self.similarity_threshold = similarity_threshold
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, no punctuation, etc.)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def find_best_match(
        self,
        segment_text: str,
        lyric_lines: List[str]
    ) -> Tuple[Optional[int], int]:
        """
        Find the best matching lyric line for a transcription segment.
        
        Args:
            segment_text: Text from the transcription segment
            lyric_lines: List of lyric lines to match against
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        segment_norm = self.normalize_text(segment_text)
        
        best_index = None
        best_score = 0
        
        for i, lyric_line in enumerate(lyric_lines):
            lyric_norm = self.normalize_text(lyric_line)
            
            # Use multiple similarity metrics and take the best
            ratio = fuzz.ratio(segment_norm, lyric_norm)
            partial_ratio = fuzz.partial_ratio(segment_norm, lyric_norm)
            token_sort_ratio = fuzz.token_sort_ratio(segment_norm, lyric_norm)
            
            score = max(ratio, partial_ratio, token_sort_ratio)
            
            if score > best_score:
                best_score = score
                best_index = i
        
        if best_score >= self.similarity_threshold:
            return best_index, best_score
        else:
            return None, best_score
    
    def align_segments(
        self,
        segments: List[Dict[str, Any]],
        lyrics: str
    ) -> List[Dict[str, Any]]:
        """
        Align transcription segments with lyric lines.
        
        Args:
            segments: List of transcription segments with timestamps
            lyrics: Full lyrics text
            
        Returns:
            List of aligned segments with matched lyrics
        """
        # Split lyrics into lines
        lyric_lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Remove section headers like [Verse], [Chorus], etc.
        lyric_lines = [
            line for line in lyric_lines 
            if not (line.startswith('[') and line.endswith(']'))
        ]
        
        aligned_segments = []
        used_lyrics = set()
        
        for segment in segments:
            segment_text = segment.get('text', '')
            
            if not segment_text.strip():
                continue
            
            # Find the best matching lyric line
            best_idx, score = self.find_best_match(segment_text, lyric_lines)
            
            aligned_segment = {
                'start': segment.get('start'),
                'end': segment.get('end'),
                'transcription': segment_text,
                'similarity_score': score
            }
            
            if best_idx is not None and best_idx not in used_lyrics:
                aligned_segment['lyrics'] = lyric_lines[best_idx]
                aligned_segment['matched'] = True
                used_lyrics.add(best_idx)
            else:
                aligned_segment['lyrics'] = segment_text
                aligned_segment['matched'] = False
            
            aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def align(
        self,
        transcript: Dict[str, Any],
        lyrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Align a full transcript with lyrics data.
        
        Args:
            transcript: Transcript data from the transcriber
            lyrics_data: Lyrics data from the scraper
            
        Returns:
            Aligned data with timestamps and corrected lyrics
        """
        segments = transcript.get('segments', [])
        lyrics_text = lyrics_data.get('lyrics', '')
        
        if not lyrics_text:
            # If no lyrics available, return transcript as-is
            return {
                'filename': transcript.get('filename'),
                'title': lyrics_data.get('title', 'Unknown'),
                'artist': lyrics_data.get('artist', 'Unknown'),
                'aligned': False,
                'segments': segments,
                'message': 'No lyrics available for alignment'
            }
        
        aligned_segments = self.align_segments(segments, lyrics_text)
        
        # Calculate alignment statistics
        total_segments = len(aligned_segments)
        matched_segments = sum(1 for s in aligned_segments if s.get('matched', False))
        avg_score = sum(s.get('similarity_score', 0) for s in aligned_segments) / max(total_segments, 1)
        
        return {
            'filename': transcript.get('filename'),
            'title': lyrics_data.get('title', 'Unknown'),
            'artist': lyrics_data.get('artist', 'Unknown'),
            'url': lyrics_data.get('url'),
            'aligned': True,
            'segments': aligned_segments,
            'statistics': {
                'total_segments': total_segments,
                'matched_segments': matched_segments,
                'match_rate': matched_segments / max(total_segments, 1),
                'average_similarity': avg_score
            }
        }
    
    def get_timed_lyrics(self, aligned_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract timed lyrics from aligned data.
        
        Args:
            aligned_data: Output from the align() method
            
        Returns:
            List of lyrics with timestamps
        """
        timed_lyrics = []
        
        for segment in aligned_data.get('segments', []):
            if segment.get('matched', False):
                timed_lyrics.append({
                    'time': segment['start'],
                    'duration': segment['end'] - segment['start'],
                    'text': segment['lyrics']
                })
        
        return timed_lyrics
    
    def export_lrc(self, aligned_data: Dict[str, Any]) -> str:
        """
        Export aligned lyrics to LRC format (karaoke lyrics).
        
        Args:
            aligned_data: Output from the align() method
            
        Returns:
            LRC formatted string
        """
        lines = [
            f"[ar:{aligned_data.get('artist', 'Unknown')}]",
            f"[ti:{aligned_data.get('title', 'Unknown')}]",
            "[by:RVC Artist - AI Generated]",
            ""
        ]
        
        for segment in aligned_data.get('segments', []):
            if segment.get('matched', False):
                start = segment['start']
                minutes = int(start // 60)
                seconds = start % 60
                text = segment['lyrics']
                
                lines.append(f"[{minutes:02d}:{seconds:05.2f}]{text}")
        
        return '\n'.join(lines)
