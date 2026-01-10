"""
YouTube Downloader Service
==========================
Downloads audio from YouTube playlists using yt-dlp.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
import re


class YouTubeDownloader:
    """Downloads audio from YouTube videos and playlists."""
    
    def __init__(self, output_dir: str = "./data/audio/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # yt-dlp options for audio extraction
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'ignoreerrors': True,
            'no_warnings': False,
            'extract_flat': False,
            'writethumbnail': True,
            'writeinfojson': True,
        }
    
    def get_playlist_info(self, playlist_url: str) -> Dict[str, Any]:
        """Get information about a playlist without downloading."""
        try:
            # Use yt-dlp to extract playlist info
            result = subprocess.run(
                [
                    'yt-dlp',
                    '--flat-playlist',
                    '--dump-json',
                    playlist_url
                ],
                capture_output=True,
                text=True
            )
            
            videos = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        video_info = json.loads(line)
                        videos.append({
                            'id': video_info.get('id'),
                            'title': video_info.get('title'),
                            'url': video_info.get('url') or f"https://youtube.com/watch?v={video_info.get('id')}",
                            'duration': video_info.get('duration')
                        })
                    except json.JSONDecodeError:
                        continue
            
            return {
                'count': len(videos),
                'videos': videos
            }
            
        except Exception as e:
            print(f"Error getting playlist info: {e}")
            return {'count': 0, 'videos': []}
    
    def download_video(self, video_url: str) -> Optional[str]:
        """Download a single video's audio."""
        try:
            result = subprocess.run(
                [
                    'yt-dlp',
                    '-x',  # Extract audio
                    '--audio-format', 'mp3',
                    '--audio-quality', '0',  # Best quality
                    '-o', str(self.output_dir / '%(title)s.%(ext)s'),
                    '--write-info-json',
                    '--no-playlist',
                    video_url
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Try to find the downloaded file
                # Parse the output to find the filename
                for line in result.stdout.split('\n'):
                    if '[ExtractAudio]' in line and 'Destination:' in line:
                        match = re.search(r'Destination: (.+\.mp3)', line)
                        if match:
                            return match.group(1)
                
                # Fallback: return most recently modified mp3
                mp3_files = list(self.output_dir.glob('*.mp3'))
                if mp3_files:
                    return str(max(mp3_files, key=os.path.getmtime))
            
            return None
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    
    def download_playlist(
        self,
        playlist_url: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Download all videos from a playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            progress_callback: Optional callback(current, total, title)
        
        Returns:
            Dictionary with download results
        """
        # First, get playlist info
        playlist_info = self.get_playlist_info(playlist_url)
        videos = playlist_info['videos']
        total = len(videos)
        
        if total == 0:
            return {
                'downloaded': 0,
                'failed': 0,
                'total': 0,
                'files': []
            }
        
        downloaded_files = []
        failed = 0
        
        for i, video in enumerate(videos):
            title = video.get('title', 'Unknown')
            video_url = video.get('url')
            
            if progress_callback:
                progress_callback(i, total, title)
            
            print(f"Downloading {i + 1}/{total}: {title}")
            
            result = self.download_video(video_url)
            
            if result:
                downloaded_files.append(result)
            else:
                failed += 1
                print(f"Failed to download: {title}")
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return {
            'downloaded': len(downloaded_files),
            'failed': failed,
            'total': total,
            'files': downloaded_files
        }
    
    def get_downloaded_files(self) -> List[Dict[str, Any]]:
        """List all downloaded audio files."""
        files = []
        
        for ext in ['mp3', 'wav', 'm4a', 'webm']:
            for f in self.output_dir.glob(f'*.{ext}'):
                info_file = f.with_suffix('.info.json')
                info = {}
                
                if info_file.exists():
                    try:
                        with open(info_file) as jf:
                            info = json.load(jf)
                    except:
                        pass
                
                files.append({
                    'path': str(f),
                    'filename': f.name,
                    'title': info.get('title', f.stem),
                    'artist': info.get('artist') or info.get('uploader'),
                    'duration': info.get('duration'),
                    'size': f.stat().st_size
                })
        
        return files


if __name__ == "__main__":
    # Test the downloader
    downloader = YouTubeDownloader()
    
    # Test with the provided playlist
    playlist_url = "https://youtube.com/playlist?list=PLmxndgxyabSoy7Rrz8osrT02DYPyXBkUw"
    
    print("Getting playlist info...")
    info = downloader.get_playlist_info(playlist_url)
    print(f"Found {info['count']} videos in playlist")
    
    for video in info['videos'][:5]:  # Show first 5
        print(f"  - {video['title']}")
