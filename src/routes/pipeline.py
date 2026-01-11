"""
Pipeline Routes - Orchestrates the music generation pipeline
"""

from flask import Blueprint, jsonify, request
import os
import json
import threading
from pathlib import Path
from datetime import datetime

pipeline_bp = Blueprint('pipeline', __name__)

# Get project root
ROOT_DIR = Path(__file__).parent.parent.parent

# Pipeline state
pipeline_state = {
    'status': 'idle',
    'current_step': None,
    'progress': 0,
    'logs': [],
    'error': None,
    'started_at': None,
    'completed_at': None
}

def log_message(message):
    """Add a log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pipeline_state['logs'].append(f"[{timestamp}] {message}")
    print(f"[Pipeline] {message}")


@pipeline_bp.route('/status', methods=['GET'])
def get_status():
    """Get current pipeline status."""
    return jsonify(pipeline_state)


@pipeline_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get pipeline logs."""
    return jsonify({'logs': pipeline_state['logs']})


@pipeline_bp.route('/clear-logs', methods=['POST'])
def clear_logs():
    """Clear pipeline logs."""
    pipeline_state['logs'] = []
    return jsonify({'status': 'cleared'})


@pipeline_bp.route('/download', methods=['POST'])
def start_download():
    """Start downloading songs from YouTube playlist."""
    from src.services.youtube_downloader import YouTubeDownloader
    
    data = request.json or {}
    playlist_url = data.get('playlist_url') or os.getenv('YOUTUBE_PLAYLIST_URL')
    
    if not playlist_url:
        return jsonify({'error': 'No playlist URL provided'}), 400
    
    def download_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'download'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            downloader = YouTubeDownloader(
                output_dir=str(ROOT_DIR / "data" / "audio" / "raw")
            )
            
            log_message(f"Starting download from: {playlist_url}")
            
            def progress_callback(current, total, title):
                pipeline_state['progress'] = int((current / total) * 100)
                log_message(f"Downloaded {current}/{total}: {title}")
            
            result = downloader.download_playlist(
                playlist_url,
                progress_callback=progress_callback
            )
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Download complete! {result['downloaded']} songs downloaded.")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during download: {str(e)}")
    
    thread = threading.Thread(target=download_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Download started in background'})


@pipeline_bp.route('/transcribe', methods=['POST'])
def start_transcription():
    """Start transcribing downloaded songs."""
    from src.services.transcriber import AudioTranscriber
    
    def transcribe_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'transcribe'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            transcriber = AudioTranscriber()
            audio_dir = ROOT_DIR / "data" / "audio" / "raw"
            output_dir = ROOT_DIR / "data" / "transcripts"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_files = list(audio_dir.glob("*.[mw][pa][3v]")) + \
                         list(audio_dir.glob("*.m4a")) + \
                         list(audio_dir.glob("*.webm"))
            
            total = len(audio_files)
            log_message(f"Found {total} audio files to transcribe")
            
            for i, audio_file in enumerate(audio_files):
                log_message(f"Transcribing: {audio_file.name}")
                
                result = transcriber.transcribe(str(audio_file))
                
                # Save transcript
                output_file = output_dir / f"{audio_file.stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                pipeline_state['progress'] = int(((i + 1) / total) * 100)
                log_message(f"Transcribed {i + 1}/{total}: {audio_file.name}")
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message("Transcription complete!")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during transcription: {str(e)}")
    
    thread = threading.Thread(target=transcribe_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Transcription started in background'})


@pipeline_bp.route('/scrape-lyrics', methods=['POST'])
def start_lyrics_scraping():
    """Start scraping lyrics from Genius."""
    from src.services.lyrics_scraper import LyricsScraper
    
    data = request.json or {}
    artist_name = data.get('artist_name', '')
    
    def scrape_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'scrape_lyrics'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            token = os.getenv('GENIUS_API_TOKEN')
            if not token or token == 'your_genius_api_token_here':
                raise ValueError("Genius API token not configured. Please set GENIUS_API_TOKEN in .env")
            
            scraper = LyricsScraper(token)
            audio_dir = ROOT_DIR / "data" / "audio" / "raw"
            output_dir = ROOT_DIR / "data" / "lyrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get song names from downloaded files
            audio_files = list(audio_dir.glob("*.[mw][pa][3v]")) + \
                         list(audio_dir.glob("*.m4a")) + \
                         list(audio_dir.glob("*.webm"))
            
            total = len(audio_files)
            log_message(f"Searching lyrics for {total} songs")
            
            for i, audio_file in enumerate(audio_files):
                song_title = audio_file.stem
                # Clean up common YouTube title patterns
                song_title = song_title.split(' - ')[-1] if ' - ' in song_title else song_title
                song_title = song_title.split('(')[0].strip()  # Remove parenthetical info
                
                log_message(f"Searching lyrics for: {song_title}")
                
                result = scraper.search_and_get_lyrics(
                    song_title=song_title,
                    artist_name=artist_name
                )
                
                if result:
                    output_file = output_dir / f"{audio_file.stem}.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    log_message(f"Found lyrics: {result.get('title', song_title)}")
                else:
                    log_message(f"No lyrics found for: {song_title}")
                
                pipeline_state['progress'] = int(((i + 1) / total) * 100)
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message("Lyrics scraping complete!")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during lyrics scraping: {str(e)}")
    
    thread = threading.Thread(target=scrape_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Lyrics scraping started in background'})


@pipeline_bp.route('/align', methods=['POST'])
def start_alignment():
    """Align transcriptions with official lyrics."""
    from src.services.lyrics_aligner import LyricsAligner
    
    def align_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'align'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            aligner = LyricsAligner()
            transcripts_dir = ROOT_DIR / "data" / "transcripts"
            lyrics_dir = ROOT_DIR / "data" / "lyrics"
            output_dir = ROOT_DIR / "data" / "aligned"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            transcript_files = list(transcripts_dir.glob("*.json"))
            total = len(transcript_files)
            
            log_message(f"Aligning {total} transcripts with lyrics")
            
            for i, transcript_file in enumerate(transcript_files):
                lyrics_file = lyrics_dir / transcript_file.name
                
                if lyrics_file.exists():
                    log_message(f"Aligning: {transcript_file.stem}")
                    
                    with open(transcript_file) as f:
                        transcript = json.load(f)
                    with open(lyrics_file) as f:
                        lyrics = json.load(f)
                    
                    aligned = aligner.align(transcript, lyrics)
                    
                    output_file = output_dir / transcript_file.name
                    with open(output_file, 'w') as f:
                        json.dump(aligned, f, indent=2)
                else:
                    log_message(f"No lyrics found for: {transcript_file.stem}")
                
                pipeline_state['progress'] = int(((i + 1) / total) * 100)
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message("Alignment complete!")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during alignment: {str(e)}")
    
    thread = threading.Thread(target=align_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Alignment started in background'})


@pipeline_bp.route('/analyze', methods=['POST'])
def start_analysis():
    """Analyze musical features of the songs."""
    from src.services.style_analyzer import StyleAnalyzer
    
    def analyze_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'analyze'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            analyzer = StyleAnalyzer()
            audio_dir = ROOT_DIR / "data" / "audio" / "raw"
            output_dir = ROOT_DIR / "data" / "features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_files = list(audio_dir.glob("*.[mw][pa][3v]")) + \
                         list(audio_dir.glob("*.m4a")) + \
                         list(audio_dir.glob("*.webm"))
            
            total = len(audio_files)
            log_message(f"Analyzing {total} audio files")
            
            all_features = []
            
            for i, audio_file in enumerate(audio_files):
                log_message(f"Analyzing: {audio_file.name}")
                
                features = analyzer.analyze(str(audio_file))
                features['filename'] = audio_file.name
                all_features.append(features)
                
                # Save individual features
                output_file = output_dir / f"{audio_file.stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(features, f, indent=2)
                
                pipeline_state['progress'] = int(((i + 1) / total) * 100)
            
            # Save aggregated style profile
            style_profile = analyzer.create_style_profile(all_features)
            with open(output_dir / "style_profile.json", 'w') as f:
                json.dump(style_profile, f, indent=2)
            
            log_message(f"Style profile created with {len(all_features)} songs")
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message("Analysis complete!")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during analysis: {str(e)}")
    
    thread = threading.Thread(target=analyze_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Analysis started in background'})


@pipeline_bp.route('/generate', methods=['POST'])
def start_generation():
    """Generate a new song."""
    from src.services.music_generator import MusicGenerator
    from src.services.lyrics_generator import LyricsGenerator
    
    data = request.json or {}
    prompt = data.get('prompt', '')
    duration = data.get('duration', 30)  # Duration in seconds
    artist_name = data.get('artist_name', 'Yeat')  # Default to Yeat for now
    
    def generate_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'generate'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            generator = MusicGenerator()
            lyrics_gen = LyricsGenerator()
            
            features_dir = ROOT_DIR / "data" / "features"
            transcripts_dir = ROOT_DIR / "data" / "transcripts"
            output_dir = ROOT_DIR / "output" / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Generate Lyrical Themes / Keywords
            log_message(f"Training lyrics generator for {artist_name}...")
            lyrics_gen.train_from_transcripts(str(transcripts_dir))
            
            new_lyrics = lyrics_gen.generate_lyrics(num_lines=4)
            log_message(f"--- GENERATED LYRICS FOR {artist_name} ---")
            for line in new_lyrics.split('\n'):
                log_message(f"  {line}")
            
            # Load style profile if available
            style_profile_path = features_dir / "style_profile.json"
            style_profile = None
            if style_profile_path.exists():
                with open(style_profile_path) as f:
                    style_profile = json.load(f)
                log_message("Loaded style profile for generation")
            
            log_message(f"Generating new song with prompt: {prompt or 'enhanced artist style'}")
            log_message(f"Duration: {duration} seconds")
            
            pipeline_state['progress'] = 40
            
            # Generate the music
            output_path = generator.generate(
                prompt=prompt,
                duration=duration,
                style_profile=style_profile,
                artist_name=artist_name,
                output_dir=str(output_dir)
            )
            
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Generation complete! Saved to: {output_path}")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error during generation: {str(e)}")
    
    thread = threading.Thread(target=generate_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Generation started in background'})


@pipeline_bp.route('/run-full', methods=['POST'])
def run_full_pipeline():
    """Run the complete pipeline from download to generation."""
    data = request.json or {}
    
    def full_pipeline_task():
        pipeline_state['status'] = 'running'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        pipeline_state['logs'] = []
        
        steps = [
            ('download', 'Downloading songs from YouTube'),
            ('transcribe', 'Transcribing audio'),
            ('scrape_lyrics', 'Scraping lyrics from Genius'),
            ('align', 'Aligning lyrics with transcriptions'),
            ('analyze', 'Analyzing musical style'),
            ('generate', 'Generating new song')
        ]
        
        try:
            from src.services.youtube_downloader import YouTubeDownloader
            from src.services.transcriber import AudioTranscriber
            from src.services.lyrics_scraper import LyricsScraper
            from src.services.lyrics_aligner import LyricsAligner
            from src.services.style_analyzer import StyleAnalyzer
            from src.services.music_generator import MusicGenerator
            
            playlist_url = data.get('playlist_url') or os.getenv('YOUTUBE_PLAYLIST_URL')
            artist_name = data.get('artist_name', '')
            prompt = data.get('prompt', '')
            duration = data.get('duration', 30)
            
            # Step 1: Download
            pipeline_state['current_step'] = 'download'
            log_message("Step 1/6: Downloading songs from YouTube...")
            
            downloader = YouTubeDownloader(str(ROOT_DIR / "data" / "audio" / "raw"))
            download_result = downloader.download_playlist(playlist_url)
            log_message(f"Downloaded {download_result['downloaded']} songs")
            pipeline_state['progress'] = 15
            
            # Step 2: Transcribe
            pipeline_state['current_step'] = 'transcribe'
            log_message("Step 2/6: Transcribing audio...")
            
            transcriber = AudioTranscriber()
            audio_dir = ROOT_DIR / "data" / "audio" / "raw"
            output_dir = ROOT_DIR / "data" / "transcripts"
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")) + \
                         list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.webm"))
            
            for audio_file in audio_files:
                result = transcriber.transcribe(str(audio_file))
                with open(output_dir / f"{audio_file.stem}.json", 'w') as f:
                    json.dump(result, f, indent=2)
            
            log_message(f"Transcribed {len(audio_files)} files")
            pipeline_state['progress'] = 30
            
            # Step 3: Scrape lyrics
            pipeline_state['current_step'] = 'scrape_lyrics'
            log_message("Step 3/6: Scraping lyrics from Genius...")
            
            token = os.getenv('GENIUS_API_TOKEN')
            if token and token != 'your_genius_api_token_here':
                scraper = LyricsScraper(token)
                lyrics_dir = ROOT_DIR / "data" / "lyrics"
                lyrics_dir.mkdir(parents=True, exist_ok=True)
                
                for audio_file in audio_files:
                    song_title = audio_file.stem.split(' - ')[-1].split('(')[0].strip()
                    result = scraper.search_and_get_lyrics(song_title, artist_name)
                    if result:
                        with open(lyrics_dir / f"{audio_file.stem}.json", 'w') as f:
                            json.dump(result, f, indent=2)
            else:
                log_message("Skipping lyrics scraping - no Genius API token")
            
            pipeline_state['progress'] = 45
            
            # Step 4: Align
            pipeline_state['current_step'] = 'align'
            log_message("Step 4/6: Aligning lyrics with transcriptions...")
            
            aligner = LyricsAligner()
            transcripts_dir = ROOT_DIR / "data" / "transcripts"
            lyrics_dir = ROOT_DIR / "data" / "lyrics"
            aligned_dir = ROOT_DIR / "data" / "aligned"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            
            for transcript_file in transcripts_dir.glob("*.json"):
                lyrics_file = lyrics_dir / transcript_file.name
                if lyrics_file.exists():
                    with open(transcript_file) as f:
                        transcript = json.load(f)
                    with open(lyrics_file) as f:
                        lyrics = json.load(f)
                    
                    aligned = aligner.align(transcript, lyrics)
                    with open(aligned_dir / transcript_file.name, 'w') as f:
                        json.dump(aligned, f, indent=2)
            
            pipeline_state['progress'] = 60
            
            # Step 5: Analyze
            pipeline_state['current_step'] = 'analyze'
            log_message("Step 5/6: Analyzing musical style...")
            
            analyzer = StyleAnalyzer()
            features_dir = ROOT_DIR / "data" / "features"
            features_dir.mkdir(parents=True, exist_ok=True)
            
            all_features = []
            for audio_file in audio_files:
                features = analyzer.analyze(str(audio_file))
                features['filename'] = audio_file.name
                all_features.append(features)
                with open(features_dir / f"{audio_file.stem}.json", 'w') as f:
                    json.dump(features, f, indent=2)
            
            style_profile = analyzer.create_style_profile(all_features)
            with open(features_dir / "style_profile.json", 'w') as f:
                json.dump(style_profile, f, indent=2)
            
            pipeline_state['progress'] = 80
            
            # Step 6: Generate
            pipeline_state['current_step'] = 'generate'
            log_message("Step 6/6: Generating new song...")
            
            # Use LyricsGenerator to show what the artist would say
            try:
                from src.services.lyrics_generator import LyricsGenerator
                lyrics_gen = LyricsGenerator()
                transcripts_dir = ROOT_DIR / "data" / "transcripts"
                log_message(f"Training lyrics generator for {artist_name}...")
                lyrics_gen.train_from_transcripts(str(transcripts_dir))
                new_lyrics = lyrics_gen.generate_lyrics(num_lines=4)
                log_message(f"--- GENERATED LYRICS FOR {artist_name} ---")
                for line in new_lyrics.split('\n'):
                    log_message(f"  {line}")
            except Exception as e:
                log_message(f"Could not generate lyrics: {str(e)}")

            generator = MusicGenerator()
            output_path = generator.generate(
                prompt=prompt,
                duration=duration,
                style_profile=style_profile,
                artist_name=artist_name,
                output_dir=str(ROOT_DIR / "output" / "generated")
            )
            
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Pipeline complete! Generated song saved to: {output_path}")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Pipeline error: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=full_pipeline_task)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Full pipeline started in background'})
