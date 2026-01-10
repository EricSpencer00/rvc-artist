"""
Flask Application Factory
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

def create_app():
    """Create and configure the Flask application."""
    
    # Get the project root directory (src/app.py -> src -> rvc-artist)
    root_dir = Path(__file__).parent.parent
    template_dir = root_dir / "frontend" / "templates"
    static_dir = root_dir / "frontend" / "static"
    
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )
    
    # Enable CORS
    CORS(app)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
    
    # Register blueprints
    from src.routes.api import api_bp
    from src.routes.pipeline import pipeline_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(pipeline_bp, url_prefix='/pipeline')
    
    # Main routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/status')
    def status():
        return jsonify({
            'status': 'running',
            'version': '1.0.0',
            'components': {
                'youtube_downloader': True,
                'transcriber': True,
                'lyrics_scraper': True,
                'music_generator': True
            }
        })
    
    @app.route('/output/<path:filename>')
    def serve_output(filename):
        output_dir = root_dir / "output"
        return send_from_directory(str(output_dir), filename)
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app
