"""
Main API Routes
"""

from flask import Blueprint, jsonify, request
import os
from pathlib import Path

api_bp = Blueprint('api', __name__)

# Get project root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

@api_bp.route('/songs', methods=['GET'])
def list_songs():
    """List all downloaded songs."""
    audio_dir = ROOT_DIR / "data" / "audio" / "raw"
    songs = []
    
    if audio_dir.exists():
        for f in audio_dir.iterdir():
            if f.suffix in ['.mp3', '.wav', '.m4a', '.webm']:
                songs.append({
                    'name': f.stem,
                    'filename': f.name,
                    'path': str(f),
                    'size': f.stat().st_size
                })
    
    return jsonify({'songs': songs, 'count': len(songs)})


@api_bp.route('/transcripts', methods=['GET'])
def list_transcripts():
    """List all transcriptions."""
    transcripts_dir = ROOT_DIR / "data" / "transcripts"
    transcripts = []
    
    if transcripts_dir.exists():
        for f in transcripts_dir.glob("*.json"):
            transcripts.append({
                'name': f.stem,
                'filename': f.name
            })
    
    return jsonify({'transcripts': transcripts, 'count': len(transcripts)})


@api_bp.route('/lyrics', methods=['GET'])
def list_lyrics():
    """List all scraped lyrics."""
    lyrics_dir = ROOT_DIR / "data" / "lyrics"
    lyrics = []
    
    if lyrics_dir.exists():
        for f in lyrics_dir.glob("*.json"):
            lyrics.append({
                'name': f.stem,
                'filename': f.name
            })
    
    return jsonify({'lyrics': lyrics, 'count': len(lyrics)})


@api_bp.route('/generated', methods=['GET'])
def list_generated():
    """List all generated songs."""
    output_dir = ROOT_DIR / "output" / "generated"
    generated = []
    
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.suffix in ['.mp3', '.wav']:
                generated.append({
                    'name': f.stem,
                    'filename': f.name,
                    'url': f'/output/generated/{f.name}'
                })
    
    return jsonify({'generated': generated, 'count': len(generated)})


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        'genius_configured': bool(os.getenv('GENIUS_API_TOKEN')),
        'playlist_url': os.getenv('YOUTUBE_PLAYLIST_URL', ''),
        'data_dir': str(ROOT_DIR / "data"),
        'output_dir': str(ROOT_DIR / "output")
    })


@api_bp.route('/config', methods=['POST'])
def update_config():
    """Update configuration."""
    data = request.json
    
    # Update environment variables in memory
    if 'genius_token' in data:
        os.environ['GENIUS_API_TOKEN'] = data['genius_token']
    
    if 'playlist_url' in data:
        os.environ['YOUTUBE_PLAYLIST_URL'] = data['playlist_url']
    
    return jsonify({'status': 'updated'})
