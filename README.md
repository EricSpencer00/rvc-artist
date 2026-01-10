# RVC Artist - AI Music Generation Pipeline

🎵 An AI-powered tool for creating music in an artist's style by analyzing their existing catalog and generating new songs.

## Features

- **YouTube Playlist Downloader**: Download audio from YouTube playlists
- **Audio Transcription**: Uses OpenAI Whisper for accurate vocal transcription
- **Lyrics Scraping**: Fetches official lyrics from Genius
- **Lyrics Alignment**: Aligns transcriptions with official lyrics using fuzzy matching
- **Style Analysis**: Extracts musical features (tempo, key, energy, spectral characteristics)
- **Music Generation**: Creates new songs using Meta's AudioCraft/MusicGen based on learned style

## Architecture

```
RVC Artist Pipeline
├── Download → YouTube audio extraction
├── Transcribe → Whisper-based transcription
├── Scrape Lyrics → Genius API integration
├── Align → Fuzzy matching alignment
├── Analyze → Librosa feature extraction
└── Generate → AudioCraft music generation
```

## Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg (for audio processing)
- 8GB+ RAM recommended
- GPU optional but recommended for music generation

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd rvc-artist
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GENIUS_API_TOKEN=your_genius_api_token_here
YOUTUBE_PLAYLIST_URL=https://youtube.com/playlist?list=YOUR_PLAYLIST_ID
```

Get a Genius API token from: https://genius.com/api-clients

5. **Run the application:**
```bash
python app.py
```

6. **Open your browser:**
Navigate to `http://localhost:5000`

## Usage

### Web Interface

The easiest way to use RVC Artist is through the web interface:

1. Open `http://localhost:5000` in your browser
2. Enter a YouTube playlist URL
3. Enter the artist name for lyrics matching
4. Click "Run Full Pipeline" or run individual steps
5. Monitor progress in real-time
6. Generate new music with custom prompts

### API Endpoints

#### Pipeline Operations

**Start Download:**
```bash
curl -X POST http://localhost:5000/pipeline/download \
  -H "Content-Type: application/json" \
  -d '{"playlist_url": "https://youtube.com/playlist?list=..."}'
```

**Start Transcription:**
```bash
curl -X POST http://localhost:5000/pipeline/transcribe
```

**Scrape Lyrics:**
```bash
curl -X POST http://localhost:5000/pipeline/scrape-lyrics \
  -H "Content-Type: application/json" \
  -d '{"artist_name": "Artist Name"}'
```

**Align Lyrics:**
```bash
curl -X POST http://localhost:5000/pipeline/align
```

**Analyze Style:**
```bash
curl -X POST http://localhost:5000/pipeline/analyze
```

**Generate Music:**
```bash
curl -X POST http://localhost:5000/pipeline/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "upbeat pop song", "duration": 30}'
```

**Run Full Pipeline:**
```bash
curl -X POST http://localhost:5000/pipeline/run-full \
  -H "Content-Type: application/json" \
  -d '{
    "playlist_url": "https://youtube.com/playlist?list=...",
    "artist_name": "Artist Name",
    "prompt": "energetic pop song",
    "duration": 30
  }'
```

#### Data Endpoints

**List Downloaded Songs:**
```bash
curl http://localhost:5000/api/songs
```

**List Transcripts:**
```bash
curl http://localhost:5000/api/transcripts
```

**List Lyrics:**
```bash
curl http://localhost:5000/api/lyrics
```

**List Generated Songs:**
```bash
curl http://localhost:5000/api/generated
```

**Get Pipeline Status:**
```bash
curl http://localhost:5000/pipeline/status
```

## Project Structure

```
rvc-artist/
├── app.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── frontend/
│   ├── templates/
│   │   └── index.html         # Web UI
│   └── static/                # Static assets
├── src/
│   ├── app.py                 # Flask application factory
│   ├── routes/
│   │   ├── api.py            # API routes
│   │   └── pipeline.py       # Pipeline orchestration
│   └── services/
│       ├── youtube_downloader.py   # YouTube audio download
│       ├── transcriber.py          # Whisper transcription
│       ├── lyrics_scraper.py       # Genius lyrics scraping
│       ├── lyrics_aligner.py       # Lyrics alignment
│       ├── style_analyzer.py       # Musical feature analysis
│       └── music_generator.py      # MusicGen generation
├── data/                      # Generated data (gitignored)
│   ├── audio/
│   │   ├── raw/              # Downloaded audio
│   │   ├── vocals/           # Separated vocals
│   │   └── instrumentals/    # Separated instrumentals
│   ├── transcripts/          # Whisper transcriptions
│   ├── lyrics/               # Scraped lyrics
│   ├── aligned/              # Aligned lyrics + timestamps
│   └── features/             # Extracted musical features
├── models/                    # Model checkpoints (gitignored)
└── output/                    # Generated songs (gitignored)
    └── generated/
```

## Services Documentation

### YouTubeDownloader

Downloads audio from YouTube videos and playlists.

```python
from src.services.youtube_downloader import YouTubeDownloader

downloader = YouTubeDownloader(output_dir="./data/audio/raw")
result = downloader.download_playlist(playlist_url)
```

### AudioTranscriber

Transcribes audio using OpenAI Whisper.

```python
from src.services.transcriber import AudioTranscriber

transcriber = AudioTranscriber(model_size="base")
result = transcriber.transcribe("audio.mp3")
```

### LyricsScraper

Scrapes lyrics from Genius.

```python
from src.services.lyrics_scraper import LyricsScraper

scraper = LyricsScraper(api_token="your_token")
lyrics = scraper.search_and_get_lyrics(
    song_title="Song Name",
    artist_name="Artist Name"
)
```

### LyricsAligner

Aligns transcriptions with official lyrics.

```python
from src.services.lyrics_aligner import LyricsAligner

aligner = LyricsAligner()
aligned = aligner.align(transcript, lyrics_data)
```

### StyleAnalyzer

Analyzes musical features using librosa.

```python
from src.services.style_analyzer import StyleAnalyzer

analyzer = StyleAnalyzer()
features = analyzer.analyze("audio.mp3")
profile = analyzer.create_style_profile([features1, features2])
```

### MusicGenerator

Generates music using AudioCraft/MusicGen.

```python
from src.services.music_generator import MusicGenerator

generator = MusicGenerator(model_size="small")
output_path = generator.generate(
    prompt="upbeat pop song",
    duration=30,
    style_profile=profile
)
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GENIUS_API_TOKEN` | Genius API token for lyrics | Yes |
| `YOUTUBE_PLAYLIST_URL` | Default playlist URL | No |
| `FLASK_HOST` | Server host (default: 0.0.0.0) | No |
| `FLASK_PORT` | Server port (default: 5000) | No |
| `FLASK_DEBUG` | Debug mode (default: True) | No |

### Model Sizes

**Whisper Models:**
- `tiny` - Fastest, least accurate (~1GB)
- `base` - Good balance (default, ~1GB)
- `small` - Better accuracy (~2GB)
- `medium` - High accuracy (~5GB)
- `large` - Best accuracy (~10GB)

**MusicGen Models:**
- `small` - Fastest generation (300M params)
- `medium` - Better quality (1.5B params)
- `large` - Best quality (3.3B params)
- `melody` - Supports melody conditioning

## Troubleshooting

### FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### CUDA/GPU issues
```bash
# For CPU-only (slower but works everywhere)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory errors
- Use smaller models (Whisper `tiny` or `base`, MusicGen `small`)
- Reduce batch sizes
- Process fewer songs at once

### Genius API errors
- Verify your API token is correct
- Check rate limits (typically 1 request/second)
- Ensure artist/song names are spelled correctly

## Performance Tips

1. **Use GPU**: Significantly faster for transcription and generation
2. **Adjust model sizes**: Trade accuracy for speed as needed
3. **Batch processing**: Process multiple files together when possible
4. **Cache results**: Results are saved to avoid reprocessing

## Legal & Ethics

- **Copyright**: Only use with content you have rights to analyze
- **Genius Terms**: Comply with Genius API terms of service
- **YouTube Terms**: Respect YouTube's terms of service
- **Generated Content**: AI-generated music may have licensing implications

## Dependencies

### Core
- Flask - Web framework
- yt-dlp - YouTube downloading
- openai-whisper - Audio transcription
- librosa - Audio analysis
- audiocraft - Music generation

### Audio Processing
- pydub - Audio manipulation
- soundfile - Audio I/O
- scipy - Scientific computing

### AI/ML
- torch - PyTorch framework
- transformers - HuggingFace models
- accelerate - Model optimization

### Utilities
- lyricsgenius - Genius API client
- beautifulsoup4 - HTML parsing
- fuzzywuzzy - Fuzzy string matching

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes. Respect copyright and licensing of source materials.

## Acknowledgments

- OpenAI Whisper for transcription
- Meta AudioCraft for music generation
- Genius for lyrics API
- yt-dlp for YouTube downloading
- librosa for audio analysis

## Support

For issues and questions:
- Check existing issues on GitHub
- Review documentation above
- Create a new issue with details

---

Built with ❤️ for music AI enthusiasts
