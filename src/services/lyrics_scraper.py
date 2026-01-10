"""
Lyrics Scraper Service
======================
Scrapes lyrics from Genius using their API.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from bs4 import BeautifulSoup


class LyricsScraper:
    """Scrapes lyrics from Genius."""
    
    BASE_URL = "https://api.genius.com"
    
    def __init__(self, api_token: str):
        """
        Initialize the scraper with Genius API token.
        
        Get your token at: https://genius.com/api-clients
        """
        self.api_token = api_token
        self.headers = {
            'Authorization': f'Bearer {api_token}'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting
        self.request_delay = 0.5  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        per_page: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for songs on Genius.
        
        Args:
            query: Search query (song title, artist, or both)
            per_page: Number of results to return
        
        Returns:
            List of song search results
        """
        self._rate_limit()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params={'q': query, 'per_page': per_page}
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for hit in data.get('response', {}).get('hits', []):
                song = hit.get('result', {})
                results.append({
                    'id': song.get('id'),
                    'title': song.get('title'),
                    'title_with_featured': song.get('title_with_featured'),
                    'artist': song.get('primary_artist', {}).get('name'),
                    'url': song.get('url'),
                    'thumbnail': song.get('song_art_image_thumbnail_url'),
                    'full_title': song.get('full_title')
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching Genius: {e}")
            return []
    
    def get_song(self, song_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a song.
        
        Args:
            song_id: Genius song ID
        
        Returns:
            Song details including metadata
        """
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.BASE_URL}/songs/{song_id}")
            response.raise_for_status()
            data = response.json()
            
            song = data.get('response', {}).get('song', {})
            
            return {
                'id': song.get('id'),
                'title': song.get('title'),
                'artist': song.get('primary_artist', {}).get('name'),
                'album': song.get('album', {}).get('name') if song.get('album') else None,
                'release_date': song.get('release_date_for_display'),
                'url': song.get('url'),
                'thumbnail': song.get('song_art_image_url'),
                'description': song.get('description', {}).get('plain') if isinstance(song.get('description'), dict) else None,
                'producer_artists': [p.get('name') for p in song.get('producer_artists', [])],
                'writer_artists': [w.get('name') for w in song.get('writer_artists', [])],
                'featured_artists': [f.get('name') for f in song.get('featured_artists', [])]
            }
            
        except Exception as e:
            print(f"Error getting song {song_id}: {e}")
            return None
    
    def get_lyrics(self, song_url: str) -> Optional[str]:
        """
        Scrape lyrics from a Genius song page.
        
        Args:
            song_url: URL of the Genius song page
        
        Returns:
            Lyrics text
        """
        self._rate_limit()
        
        try:
            # Genius lyrics aren't available via API, need to scrape
            response = requests.get(song_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find lyrics containers - Genius uses data-lyrics-container
            lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
            
            if not lyrics_containers:
                # Fallback to older class-based selectors
                lyrics_containers = soup.find_all('div', class_=re.compile(r'Lyrics__Container'))
            
            if not lyrics_containers:
                # Another fallback
                lyrics_div = soup.find('div', class_='lyrics')
                if lyrics_div:
                    return lyrics_div.get_text(separator='\n').strip()
                return None
            
            lyrics_parts = []
            for container in lyrics_containers:
                # Process the container to preserve structure
                text = self._extract_text_from_element(container)
                lyrics_parts.append(text)
            
            lyrics = '\n'.join(lyrics_parts)
            
            # Clean up the lyrics
            lyrics = self._clean_lyrics(lyrics)
            
            return lyrics
            
        except Exception as e:
            print(f"Error scraping lyrics from {song_url}: {e}")
            return None
    
    def _extract_text_from_element(self, element) -> str:
        """Extract text while preserving line breaks."""
        text_parts = []
        
        for child in element.children:
            if isinstance(child, str):
                text_parts.append(child)
            elif child.name == 'br':
                text_parts.append('\n')
            elif child.name == 'a':
                # Links often contain annotated lyrics
                text_parts.append(child.get_text())
            elif hasattr(child, 'get_text'):
                # Recursively get text from child elements
                text_parts.append(self._extract_text_from_element(child))
        
        return ''.join(text_parts)
    
    def _clean_lyrics(self, lyrics: str) -> str:
        """Clean up scraped lyrics."""
        # Remove multiple consecutive newlines
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in lyrics.split('\n')]
        lyrics = '\n'.join(lines)
        
        # Remove any remaining HTML entities
        lyrics = lyrics.replace('&amp;', '&')
        lyrics = lyrics.replace('&quot;', '"')
        lyrics = lyrics.replace('&#x27;', "'")
        lyrics = lyrics.replace('&lt;', '<')
        lyrics = lyrics.replace('&gt;', '>')
        
        return lyrics.strip()
    
    def search_and_get_lyrics(
        self,
        song_title: str,
        artist_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a song and get its lyrics.
        
        Args:
            song_title: Title of the song
            artist_name: Optional artist name for better matching
        
        Returns:
            Dictionary with song info and lyrics
        """
        # Build search query
        query = f"{artist_name} {song_title}".strip() if artist_name else song_title
        
        print(f"Searching Genius for: {query}")
        results = self.search(query, per_page=5)
        
        if not results:
            print(f"No results found for: {query}")
            return None
        
        # Try to find the best match
        best_match = None
        
        for result in results:
            # Check if artist matches (if provided)
            if artist_name:
                if artist_name.lower() in result['artist'].lower() or \
                   result['artist'].lower() in artist_name.lower():
                    best_match = result
                    break
            
            # Check if title is similar
            if song_title.lower() in result['title'].lower() or \
               result['title'].lower() in song_title.lower():
                best_match = result
                break
        
        # If no good match, use first result
        if not best_match:
            best_match = results[0]
        
        print(f"Found match: {best_match['full_title']}")
        
        # Get detailed song info
        song_info = self.get_song(best_match['id'])
        
        # Get lyrics
        lyrics = self.get_lyrics(best_match['url'])
        
        if lyrics:
            return {
                'id': best_match['id'],
                'title': best_match['title'],
                'artist': best_match['artist'],
                'full_title': best_match['full_title'],
                'url': best_match['url'],
                'lyrics': lyrics,
                'metadata': song_info,
                'sections': self._parse_sections(lyrics)
            }
        
        return None
    
    def _parse_sections(self, lyrics: str) -> List[Dict[str, Any]]:
        """
        Parse lyrics into sections (verse, chorus, bridge, etc.)
        
        Args:
            lyrics: Full lyrics text
        
        Returns:
            List of sections with type and content
        """
        sections = []
        current_section = {'type': 'intro', 'content': []}
        
        # Common section markers
        section_pattern = re.compile(
            r'\[(Verse|Chorus|Bridge|Hook|Pre-Chorus|Post-Chorus|Outro|Intro|Refrain|Interlude).*?\]',
            re.IGNORECASE
        )
        
        for line in lyrics.split('\n'):
            match = section_pattern.match(line)
            
            if match:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append(current_section)
                
                # Start new section
                section_type = match.group(1).lower()
                current_section = {
                    'type': section_type,
                    'marker': line,
                    'content': []
                }
            else:
                # Add line to current section
                if line.strip():
                    current_section['content'].append(line)
        
        # Don't forget the last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def get_artist_songs(
        self,
        artist_name: str,
        max_songs: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get songs by a specific artist.
        
        Args:
            artist_name: Name of the artist
            max_songs: Maximum number of songs to retrieve
        
        Returns:
            List of songs by the artist
        """
        self._rate_limit()
        
        try:
            # First search for the artist
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params={'q': artist_name, 'per_page': 1}
            )
            response.raise_for_status()
            data = response.json()
            
            hits = data.get('response', {}).get('hits', [])
            if not hits:
                return []
            
            # Get artist ID from first result
            artist_id = hits[0].get('result', {}).get('primary_artist', {}).get('id')
            
            if not artist_id:
                return []
            
            # Get artist's songs
            songs = []
            page = 1
            
            while len(songs) < max_songs:
                self._rate_limit()
                
                response = self.session.get(
                    f"{self.BASE_URL}/artists/{artist_id}/songs",
                    params={'page': page, 'per_page': 20, 'sort': 'popularity'}
                )
                response.raise_for_status()
                data = response.json()
                
                page_songs = data.get('response', {}).get('songs', [])
                
                if not page_songs:
                    break
                
                for song in page_songs:
                    songs.append({
                        'id': song.get('id'),
                        'title': song.get('title'),
                        'url': song.get('url'),
                        'full_title': song.get('full_title')
                    })
                    
                    if len(songs) >= max_songs:
                        break
                
                page += 1
            
            return songs
            
        except Exception as e:
            print(f"Error getting artist songs: {e}")
            return []


if __name__ == "__main__":
    # Test the scraper
    token = os.getenv('GENIUS_API_TOKEN')
    
    if not token or token == 'your_genius_api_token_here':
        print("Please set GENIUS_API_TOKEN environment variable")
        print("Get your token at: https://genius.com/api-clients")
    else:
        scraper = LyricsScraper(token)
        
        # Test search
        results = scraper.search("Bohemian Rhapsody Queen")
        if results:
            print(f"Found: {results[0]['full_title']}")
            
            # Get lyrics
            lyrics_data = scraper.search_and_get_lyrics("Bohemian Rhapsody", "Queen")
            if lyrics_data:
                print(f"\nLyrics preview:\n{lyrics_data['lyrics'][:500]}...")
