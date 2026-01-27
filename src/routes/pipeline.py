"""
Pipeline Routes - Orchestrates the music generation pipeline
Supports named style profiles, section-aware generation, and blueprint-based composition.
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


# ============================================================================
# RVC VOICE CONVERSION ROUTES
# ============================================================================

@pipeline_bp.route('/rvc/train', methods=['POST'])
def rvc_train_model():
    """Train an RVC model on voice samples."""
    from src.services.rvc_converter import RVCVoiceConverter
    
    data = request.json or {}
    voice_samples_dir = data.get('voice_samples_dir')
    model_name = data.get('model_name', 'yeat')
    
    if not voice_samples_dir:
        return jsonify({'error': 'voice_samples_dir required'}), 400
    
    voice_dir = Path(voice_samples_dir)
    if not voice_dir.exists():
        return jsonify({'error': f'Directory not found: {voice_samples_dir}'}), 404
    
    def train_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'rvc_train'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            converter = RVCVoiceConverter()
            log_message(f"Training RVC model '{model_name}' on voice samples from {voice_samples_dir}")
            
            result = converter.train_model(
                voice_samples_dir=voice_samples_dir,
                model_name=model_name,
                epochs=20,
                batch_size=8
            )
            
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"✓ Model trained: {result['message']}")
            log_message(f"Processed {result['trained_samples']} samples")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error training model: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=train_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Training RVC model "{model_name}" in background'
    })


@pipeline_bp.route('/rvc/models', methods=['GET'])
def list_rvc_models():
    """List available RVC models."""
    from src.services.rvc_converter import RVCVoiceConverter
    
    converter = RVCVoiceConverter()
    models = converter.list_models()
    
    return jsonify({
        'models': models,
        'count': len(models)
    })


@pipeline_bp.route('/rvc/convert', methods=['POST'])
def rvc_convert_voice():
    """Apply voice conversion to audio using trained RVC model."""
    from src.services.rvc_converter import RVCVoiceConverter
    from werkzeug.utils import secure_filename
    
    # Get model name and settings
    data = request.form
    model_name = data.get('model_name', 'yeat')
    pitch_shift = int(data.get('pitch_shift', 0))
    
    # Check for audio file
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    audio_filename = secure_filename(audio_file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f"{timestamp}_{audio_filename}"
    audio_path = upload_dir / audio_filename
    
    audio_file.save(str(audio_path))
    
    def convert_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'rvc_convert'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            converter = RVCVoiceConverter()
            output_dir = ROOT_DIR / "output" / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"converted_{model_name}_{timestamp}.wav"
            output_path = output_dir / output_filename
            
            log_message(f"Converting voice using model: {model_name}")
            log_message(f"Input file: {audio_filename}")
            log_message(f"Pitch shift: {pitch_shift} semitones")
            
            pipeline_state['progress'] = 20
            
            result = converter.convert_voice(
                input_audio_path=str(audio_path),
                model_name=model_name,
                output_path=str(output_path),
                pitch_shift=pitch_shift
            )
            
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"✓ Voice conversion complete!")
            log_message(f"Output: {output_filename}")
            log_message(f"Duration: {result['duration']:.2f} seconds")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error converting voice: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=convert_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Converting voice using model "{model_name}"',
        'uploaded_file': audio_filename
    })


@pipeline_bp.route('/rvc/separate-stems', methods=['POST'])
def separate_stems():
    """Separate vocal and instrumental stems from audio."""
    from src.services.stem_separator import StemSeparator
    from werkzeug.utils import secure_filename
    
    # Check for audio file
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    audio_filename = secure_filename(audio_file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f"{timestamp}_{audio_filename}"
    audio_path = upload_dir / audio_filename
    
    audio_file.save(str(audio_path))
    
    def separate_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'separate_stems'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            separator = StemSeparator()
            output_dir = ROOT_DIR / "data" / "stems" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            log_message(f"Separating stems from: {audio_filename}")
            
            pipeline_state['progress'] = 30
            
            result = separator.separate_stems(
                audio_path=str(audio_path),
                output_dir=str(output_dir),
                model='htdemucs'
            )
            
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"✓ Stem separation complete!")
            log_message(f"Vocal: {Path(result['vocal_file']).name}")
            log_message(f"Instrumental: {Path(result['instrumental_file']).name}")
            log_message(f"Duration: {result['duration']:.2f} seconds")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error separating stems: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=separate_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Separating vocal and instrumental stems',
        'uploaded_file': audio_filename
    })


@pipeline_bp.route('/rvc/full-conversion', methods=['POST'])
def rvc_full_conversion():
    """
    Full Yeat voice conversion pipeline:
    1. Separate vocals and instrumental
    2. Convert vocals to Yeat voice
    3. Mix back together
    """
    from src.services.stem_separator import StemSeparator
    from src.services.rvc_converter import RVCVoiceConverter
    from werkzeug.utils import secure_filename
    
    # Check for audio file
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    model_name = request.form.get('model_name', 'yeat')
    pitch_shift = int(request.form.get('pitch_shift', 0))
    vocal_level_db = float(request.form.get('vocal_level_db', 0.0))
    
    # Save uploaded file
    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    audio_filename = secure_filename(audio_file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f"{timestamp}_{audio_filename}"
    audio_path = upload_dir / audio_filename
    
    audio_file.save(str(audio_path))
    
    def full_conversion_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'full_conversion'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            separator = StemSeparator()
            converter = RVCVoiceConverter()
            
            output_dir = ROOT_DIR / "output" / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            stems_dir = ROOT_DIR / "data" / "stems" / timestamp
            stems_dir.mkdir(parents=True, exist_ok=True)
            
            log_message(f"Starting full Yeat voice conversion for: {audio_filename}")
            log_message(f"Model: {model_name}")
            
            # Step 1: Separate stems
            log_message("Step 1/4: Separating vocal and instrumental stems...")
            pipeline_state['progress'] = 15
            
            separation_result = separator.separate_stems(
                audio_path=str(audio_path),
                output_dir=str(stems_dir),
                model='htdemucs'
            )
            
            vocal_file = separation_result['vocal_file']
            instrumental_file = separation_result['instrumental_file']
            
            log_message(f"✓ Stems separated")
            
            # Step 2: Convert voice
            log_message("Step 2/4: Converting voice to Yeat...")
            pipeline_state['progress'] = 50
            
            converted_filename = f"converted_vocal_{timestamp}.wav"
            converted_path = stems_dir / converted_filename
            
            conversion_result = converter.convert_voice(
                input_audio_path=vocal_file,
                model_name=model_name,
                output_path=str(converted_path),
                pitch_shift=pitch_shift
            )
            
            log_message(f"✓ Voice converted")
            
            # Step 3: Mix stems
            log_message("Step 3/4: Mixing vocal and instrumental...")
            pipeline_state['progress'] = 75
            
            final_filename = f"yeat_{Path(audio_filename).stem}_{timestamp}.wav"
            final_path = output_dir / final_filename
            
            mix_result = separator.mix_stems(
                vocal_path=str(converted_path),
                instrumental_path=instrumental_file,
                output_path=str(final_path),
                vocal_level_db=vocal_level_db,
                instrumental_level_db=0.0
            )
            
            log_message(f"✓ Stems mixed")
            
            # Complete
            pipeline_state['progress'] = 100
            pipeline_state['status'] = 'completed'
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"✓ Full conversion complete!")
            log_message(f"Output: {final_filename}")
            log_message(f"Duration: {mix_result['duration']:.2f} seconds")
            log_message(f"Download: /output/generated/{final_filename}")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error in full conversion: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=full_conversion_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Converting to Yeat voice - this may take a few minutes',
        'model': model_name,
        'uploaded_file': audio_filename
    })


# ============================================================================
# LEGACY UPLOAD ROUTES (DEPRECATED)
# ============================================================================



# ============================================================================
# STYLE PROFILE MANAGEMENT ROUTES
# ============================================================================

@pipeline_bp.route('/profiles', methods=['GET'])
def list_profiles():
    """List all available style profiles."""
    from src.services.style_analyzer import StyleAnalyzer
    
    analyzer = StyleAnalyzer()
    profiles = analyzer.list_profiles(str(ROOT_DIR / "data" / "features"))
    
    return jsonify({
        'profiles': profiles,
        'count': len(profiles)
    })


@pipeline_bp.route('/profiles/<profile_name>', methods=['GET'])
def get_profile(profile_name):
    """Get a specific style profile by name."""
    from src.services.style_analyzer import StyleAnalyzer
    
    analyzer = StyleAnalyzer()
    profile = analyzer.load_named_profile(profile_name, str(ROOT_DIR / "data" / "features"))
    
    if profile:
        return jsonify(profile)
    else:
        return jsonify({'error': f'Profile "{profile_name}" not found'}), 404


@pipeline_bp.route('/profiles/create', methods=['POST'])
def create_named_profile():
    """Create a named style profile from specific audio files."""
    from src.services.style_analyzer import StyleAnalyzer
    
    data = request.json or {}
    profile_name = data.get('profile_name', 'custom')
    audio_files = data.get('audio_files', [])  # List of filenames or paths
    
    if not audio_files:
        # Use all files in raw audio directory
        audio_dir = ROOT_DIR / "data" / "audio" / "raw"
        audio_files = (
            list(audio_dir.glob("*.mp3")) +
            list(audio_dir.glob("*.wav")) +
            list(audio_dir.glob("*.m4a")) +
            list(audio_dir.glob("*.webm"))
        )
        audio_files = [str(f) for f in audio_files]
    else:
        # Resolve relative paths
        audio_dir = ROOT_DIR / "data" / "audio" / "raw"
        resolved = []
        for f in audio_files:
            if os.path.isabs(f):
                resolved.append(f)
            else:
                resolved.append(str(audio_dir / f))
        audio_files = resolved
    
    def create_profile_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'create_profile'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            analyzer = StyleAnalyzer()
            log_message(f"Creating profile '{profile_name}' from {len(audio_files)} files...")
            
            profile = analyzer.analyze_subset(
                audio_files=audio_files,
                profile_name=profile_name,
                transcripts_dir=str(ROOT_DIR / "data" / "transcripts"),
                output_dir=str(ROOT_DIR / "data" / "features")
            )
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Profile '{profile_name}' created with {profile.get('num_songs_analyzed', 0)} songs")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error creating profile: {str(e)}")
    
    thread = threading.Thread(target=create_profile_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Creating profile "{profile_name}" in background',
        'num_files': len(audio_files)
    })


# ============================================================================
# ADVANCED GENERATION ROUTES
# ============================================================================

@pipeline_bp.route('/generate-variations', methods=['POST'])
def generate_variations():
    """Generate multiple variations of a song."""
    from src.services.music_generator import MusicGenerator
    
    data = request.json or {}
    prompt = data.get('prompt', '')
    num_variations = data.get('num_variations', 3)
    duration = data.get('duration', 30)
    artist_name = data.get('artist_name', 'Yeat')
    profile_name = data.get('profile_name', None)
    
    def variations_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'generate_variations'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            from src.services.style_analyzer import StyleAnalyzer
            
            generator = MusicGenerator()
            analyzer = StyleAnalyzer()
            
            # Load style profile
            style_profile = None
            if profile_name:
                style_profile = analyzer.load_named_profile(profile_name, str(ROOT_DIR / "data" / "features"))
            else:
                default_path = ROOT_DIR / "data" / "features" / "style_profile.json"
                if default_path.exists():
                    with open(default_path) as f:
                        style_profile = json.load(f)
            
            # Build prompt if not provided
            if not prompt and style_profile:
                prompt = generator.style_to_prompt(style_profile, artist_name)
            
            log_message(f"Generating {num_variations} variations...")
            log_message(f"Prompt: {prompt[:100]}...")
            
            variations = []
            for i in range(num_variations):
                pipeline_state['progress'] = int((i / num_variations) * 100)
                log_message(f"Generating variation {i+1}/{num_variations}...")
                
                temp = 1.0 + (i * 0.15)  # Slightly vary temperature
                
                output_file = generator.generate(
                    prompt=prompt,
                    duration=duration,
                    style_profile=style_profile,
                    artist_name=artist_name,
                    output_dir=str(ROOT_DIR / "output" / "generated"),
                    temperature=temp
                )
                variations.append(output_file)
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Generated {len(variations)} variations!")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error generating variations: {str(e)}")
    
    thread = threading.Thread(target=variations_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Generating {num_variations} variations in background'
    })


@pipeline_bp.route('/generate-blueprint', methods=['POST'])
def generate_from_blueprint():
    """Generate a full song from a blueprint (section-aware generation)."""
    from src.services.music_generator import MusicGenerator
    
    data = request.json or {}
    blueprint_style = data.get('blueprint_style', 'standard')  # standard, short, trap, edm, extended
    total_duration = data.get('total_duration', 120)
    artist_name = data.get('artist_name', 'Yeat')
    profile_name = data.get('profile_name', None)
    custom_blueprint = data.get('blueprint', None)  # Allow custom section list
    
    def blueprint_task():
        pipeline_state['status'] = 'running'
        pipeline_state['current_step'] = 'generate_blueprint'
        pipeline_state['progress'] = 0
        pipeline_state['error'] = None
        pipeline_state['started_at'] = datetime.now().isoformat()
        
        try:
            from src.services.style_analyzer import StyleAnalyzer
            
            generator = MusicGenerator()
            analyzer = StyleAnalyzer()
            
            # Load style profile
            style_profile = None
            if profile_name:
                style_profile = analyzer.load_named_profile(profile_name, str(ROOT_DIR / "data" / "features"))
            else:
                default_path = ROOT_DIR / "data" / "features" / "style_profile.json"
                if default_path.exists():
                    with open(default_path) as f:
                        style_profile = json.load(f)
            
            # Create or use provided blueprint
            if custom_blueprint:
                blueprint = custom_blueprint
            else:
                blueprint = generator.create_default_blueprint(total_duration, blueprint_style)
            
            log_message(f"Generating song from blueprint (style: {blueprint_style})")
            log_message(f"Sections: {[s['type'] for s in blueprint['sections']]}")
            
            result = generator.generate_from_blueprint(
                blueprint=blueprint,
                style_profile=style_profile,
                artist_name=artist_name,
                output_dir=str(ROOT_DIR / "output" / "generated")
            )
            
            pipeline_state['status'] = 'completed'
            pipeline_state['progress'] = 100
            pipeline_state['completed_at'] = datetime.now().isoformat()
            log_message(f"Blueprint generation complete!")
            if result.get('combined_file'):
                log_message(f"Full song: {result['combined_file']}")
            
        except Exception as e:
            pipeline_state['status'] = 'error'
            pipeline_state['error'] = str(e)
            log_message(f"Error in blueprint generation: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
    
    thread = threading.Thread(target=blueprint_task)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Generating {blueprint_style} blueprint song in background'
    })


@pipeline_bp.route('/generate-lyrics', methods=['POST'])
def generate_lyrics():
    """Generate lyrics in artist style."""
    from src.services.lyrics_generator import LyricsGenerator
    from src.services.style_analyzer import StyleAnalyzer
    
    data = request.json or {}
    num_lines = data.get('num_lines', 8)
    section_type = data.get('section_type', None)  # verse, chorus, hook, etc.
    rhyme_scheme = data.get('rhyme_scheme', None)  # AABB, ABAB, etc.
    profile_name = data.get('profile_name', None)
    generate_full_song = data.get('full_song', False)
    structure = data.get('structure', None)  # For full song generation
    
    try:
        lyrics_gen = LyricsGenerator()
        analyzer = StyleAnalyzer()
        
        transcripts_dir = ROOT_DIR / "data" / "transcripts"
        lyrics_gen.train_from_transcripts(str(transcripts_dir))
        
        # Load style profile for keyword conditioning
        style_profile = None
        if profile_name:
            style_profile = analyzer.load_named_profile(profile_name, str(ROOT_DIR / "data" / "features"))
        else:
            default_path = ROOT_DIR / "data" / "features" / "style_profile.json"
            if default_path.exists():
                with open(default_path) as f:
                    style_profile = json.load(f)
        
        if generate_full_song:
            result = lyrics_gen.generate_full_song(structure=structure, style_profile=style_profile)
            return jsonify({
                'lyrics': result['full_lyrics'],
                'sections': result['sections'],
                'structure': result['structure']
            })
        elif section_type:
            lyrics = lyrics_gen.generate_section(section_type, style_profile=style_profile)
            return jsonify({
                'lyrics': lyrics,
                'section_type': section_type
            })
        else:
            keywords = style_profile.get('lyrics', {}).get('top_keywords', []) if style_profile else None
            lyrics = lyrics_gen.generate_lyrics(
                num_lines=num_lines,
                rhyme_scheme=rhyme_scheme,
                keywords=keywords
            )
            return jsonify({
                'lyrics': lyrics,
                'num_lines': num_lines,
                'rhyme_scheme': rhyme_scheme
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ORIGINAL PIPELINE ROUTES
# ============================================================================

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
