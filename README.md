# 🗣️ Lecture Voice-to-Notes & Quiz Generator

An intelligent web application that converts lecture audio recordings into comprehensive notes and interactive quizzes using AI-powered transcription and natural language processing.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [API Information](#api-information)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality

- **🎙️ Audio Transcription**
  - Upload audio files in multiple formats (MP3, WAV, M4A, FLAC, OGG)
  - Live microphone recording capability
  - Automatic language detection
  - High-accuracy transcription using Faster-Whisper
  - Automatic audio resampling to 16kHz (Whisper requirement)
  - Multi-channel to mono conversion

- **📝 Intelligent Note Generation**
  - Three note-taking styles:
    - **Cornell Notes**: Structured cue-notes-summary format
    - **Key Points/Bullet List**: Hierarchical bullet points
    - **Detailed Paragraph Summary**: Comprehensive narrative summaries
  - Powered by Google Gemini 2.5 Flash AI
  - Download notes as TXT files
  - Markdown-formatted output for better readability

- **📚 Interactive Quiz Generation**
  - Generate 3-10 multiple-choice questions
  - AI-generated questions based on lecture content
  - Randomized answer options
  - Instant feedback on answers
  - Score calculation with performance insights
  - Multiple attempt support

- **🎨 User Interface**
  - Clean, professional Streamlit interface
  - Wide layout for better content visibility
  - Tabbed navigation for organized workflow
  - Real-time progress indicators
  - Audio playback preview
  - Responsive design

## 🏗️ Architecture

```
┌─────────────────┐
│  Audio Input    │
│ (File/Live Mic) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Processing│
│  - Resampling   │
│  - Mono Convert │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Faster-Whisper  │
│  Transcription  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Transcript    │
│   (Session)     │
└────┬────────┬───┘
     │        │
     ▼        ▼
┌─────────┐ ┌──────────┐
│ Gemini  │ │  Gemini  │
│  Notes  │ │   Quiz   │
└─────────┘ └──────────┘
```

## 📦 Requirements

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection (for Gemini API)
- Modern web browser (Chrome/Firefox recommended for live recording)

### Python Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
soundfile>=0.12.0
librosa>=0.10.0
faster-whisper>=0.9.0
requests>=2.31.0
streamlit-mic-recorder>=0.0.8
```

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/lecture-voice-to-notes.git
cd lecture-voice-to-notes
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Additional System Dependencies

#### For faster-whisper (Optional but Recommended)

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**On macOS:**
```bash
brew install ffmpeg
```

**On Windows:**
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## ⚙️ Configuration

### 1. Obtain Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Set Up API Key

**Method 1: Direct Code Modification (Quick Start)**

Open the Python file and replace the API_KEY value:

```python
API_KEY = "your-actual-api-key-here"
```

**Method 2: Environment Variable (Recommended for Production)**

```bash
# On Windows (Command Prompt)
set GEMINI_API_KEY=your-actual-api-key-here

# On Windows (PowerShell)
$env:GEMINI_API_KEY="your-actual-api-key-here"

# On macOS/Linux
export GEMINI_API_KEY=your-actual-api-key-here
```

**Method 3: .env File**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-actual-api-key-here
```

Then modify the code to load from .env:

```python
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY', '')
```

### 3. Model Configuration

You can adjust the Whisper model size in the sidebar:

- **tiny**: Fastest, lowest accuracy (~1GB RAM)
- **base**: Balanced speed/accuracy (~1GB RAM) - **Default**
- **medium**: Better accuracy (~5GB RAM)
- **large-v3**: Best accuracy (~10GB RAM)

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow

#### Tab 1: Upload Audio File

1. Click "Browse files" to select an audio file
2. Preview the audio using the built-in player
3. Click "▶️ Transcribe Uploaded Lecture"
4. Wait for transcription to complete

#### Tab 2: Notes Generation

1. Select your preferred note style from the dropdown
2. Click "✨ Generate Notes"
3. Review the AI-generated notes
4. Download notes as TXT file if needed

#### Tab 3: Quiz Generation

1. Use the slider to select number of questions (3-10)
2. Click "🚀 Generate New Quiz"
3. Answer all multiple-choice questions
4. Click "Submit Quiz and See Score"
5. Review feedback and your score
6. Retry with "Start New Attempt" button

#### Tab 4: Record Live Audio

1. Grant microphone permissions when prompted
2. Click "Click to Start Recording"
3. Speak your lecture content
4. Click "Click to Stop Recording"
5. Automatic transcription begins

## 🔬 How It Works

### Audio Transcription Pipeline

1. **Audio Input**: Accept audio from file upload or live microphone
2. **Pre-processing**:
   - Read audio data using soundfile
   - Check sample rate (must be 16kHz for Whisper)
   - Resample if necessary using librosa
   - Convert to mono if stereo/multi-channel
3. **Transcription**: Use Faster-Whisper model with beam search (beam_size=5)
4. **Post-processing**: Extract text segments and detect language

### Note Generation Process

The application uses Google's Gemini 2.5 Flash model with carefully crafted prompts:

**System Instruction Format:**
```
You are a world-class AI note-taker. Your task is to process 
the following lecture transcript and generate structured, 
concise notes.

[Style-specific instructions]
```

**Styles:**
- **Cornell Notes**: Prompts for cue column, notes section, and summary
- **Bullet List**: Prompts for hierarchical key concepts
- **Paragraph**: Prompts for dense, comprehensive summary

### Quiz Generation Algorithm

1. **Prompt Engineering**: Instructs Gemini to create multiple-choice questions
2. **JSON Structure**: Enforces strict JSON format with validation
3. **Content Requirements**:
   - Exactly 4 options per question (A, B, C, D)
   - One correct answer
   - Plausible distractors
   - Coverage of different lecture topics
4. **Validation**: Checks JSON structure, required fields, and answer consistency
5. **Randomization**: Shuffles options for each question to prevent pattern memorization

## 🔑 API Information

### Faster-Whisper Models

Faster-Whisper is an optimized implementation of OpenAI's Whisper model using CTranslate2:

- **Model Source**: Pre-trained by OpenAI
- **Optimization**: CTranslate2 for faster inference
- **No Training Required**: Models are downloaded automatically on first use
- **Storage Location**: `~/.cache/huggingface/hub/`

**Model Performance:**

| Model | Size | Speed | Accuracy | RAM Usage |
|-------|------|-------|----------|-----------|
| tiny | 39M | Fastest | Basic | ~1GB |
| base | 74M | Fast | Good | ~1GB |
| medium | 769M | Moderate | Better | ~5GB |
| large-v3 | 1550M | Slow | Best | ~10GB |

### Google Gemini API

- **Model Used**: `gemini-2.5-flash-preview-05-20`
- **Capabilities**: 
  - Natural language understanding
  - Structured output generation
  - Context-aware summarization
- **Rate Limiting**: Implements exponential backoff (max 5 retries)
- **Cost**: Check [Google AI Pricing](https://ai.google.dev/pricing)

## 🐛 Troubleshooting

### Common Issues

**Issue: "faster-whisper library is not installed"**
```bash
pip install faster-whisper
pip install librosa
```

**Issue: "Gemini API key is required"**
- Verify API key is set correctly
- Check for typos or extra spaces
- Ensure API key has proper permissions

**Issue: "Could not read audio data"**
- Check audio file format is supported
- Try converting to WAV format
- Ensure file is not corrupted

**Issue: Microphone not working**
```bash
# Install mic recorder library
pip install streamlit-mic-recorder

# Grant browser microphone permissions
# Use Chrome or Firefox (recommended)
```

**Issue: Out of memory during transcription**
- Switch to a smaller Whisper model (tiny or base)
- Close other applications
- Process shorter audio files

**Issue: Quiz generation fails**
- Check internet connection
- Verify Gemini API quota
- Try reducing number of questions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Test with multiple audio formats
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Speech recognition model
- **Faster-Whisper**: Optimized Whisper implementation
- **Google Gemini**: AI language model
- **Streamlit**: Web application framework
- **Librosa**: Audio processing library

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Contact: [mandapatisirichandana@gmail.com]

## 🗺️ Roadmap

- [ ] Support for video files (extract audio)
- [ ] Multiple language UI support
- [ ] Export notes to PDF format
- [ ] Flashcard generation from notes
- [ ] Speaker diarization
- [ ] Real-time transcription display
- [ ] Integration with learning management systems
- [ ] Mobile app version

---
The app will launch automatically in your web browser at http://localhost:8501.

**Star ⭐ this repository if you find it helpful!**
