🎤 Lecture Voice-to-Notes Generator
Convert long lecture audio recordings into structured, summarized notes, interactive quizzes, and study flashcards using advanced AI models.

✨ Features
Secure User Authentication: Simulate account creation and login using Streamlit's session state.

Dual Audio Input: Supports uploading common audio file formats (.mp3, .wav, .ogg) and live microphone recording (requires streamlit-mic-recorder).

Fast Transcription: Utilizes the high-performance Faster-Whisper library for efficient, local transcription on CPU.

Intelligent Note Generation: Uses the Google Gemini API to create various note styles, including Cornell Notes, Detailed Summaries, and Key Point Bullet Lists.

Study Tools: Generates structured multiple-choice quizzes and interactive flashcard sets directly from the lecture transcript.

Interactive UI: Built using Streamlit for a simple, responsive, web-based interface.

🏛️ Architecture & Technologies
The application is structured into four key layers:

Layer	Technology	Role
Frontend/UI	Streamlit	Handles user input (upload/record), displays generated content, and manages the interactive study tools (quiz, flashcards).
Core Backend	Python	Manages session state, authentication, file handling, and orchestrates calls between transcription and generation services.
Transcription	Faster-Whisper	Converts the raw audio file into a clean, time-stamped text transcript.
Generation (LLM)	Google Gemini API	Semantic transformation of the transcript into notes, quizzes, and flashcards based on targeted prompts.

Export to Sheets
🚀 Local Deployment (Step-by-Step)
Follow these steps to set up and run the application on your local machine.

Prerequisites
Python 3.8+

FFmpeg (Crucial for audio handling). You must download FFmpeg and add its executable path to your system's environment variables (PATH).

Setup
Clone the Repository (or download files):

Bash

git clone [Your-Repo-URL]
cd lecture-voice-to-notes
Create and Activate a Virtual Environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install Required Libraries:

Bash

pip install streamlit faster-whisper requests numpy librosa soundfile streamlit-mic-recorder
Set Your API Key:
The application requires the Google Gemini API Key for notes and quiz generation. For local development, create a file named .streamlit/secrets.toml in your project root and add your key:

Ini, TOML

# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
(Note: The provided code uses a hardcoded key, but this st.secrets method is the secure best practice).

Run the Streamlit Application:
Execute the app from your terminal:

Bash

streamlit run streamlit_app.py
Access the App:
The application will automatically open in your browser at http://localhost:8501.

⚠️ Troubleshooting Common Issues
Issue	Potential Cause	Solution
ffmpeg: command not found	FFmpeg is not installed or the PATH variable is incorrect.	Install FFmpeg and ensure its bin directory is correctly added to your system's PATH.
ModuleNotFoundError: No module named 'streamlit'	Streamlit is not installed or the virtual environment is not active.	Activate your environment (source venv/bin/activate) and run pip install streamlit.
API quota exceeded	You have used up your free quota for the API key being used.	Check your billing and usage dashboard for the API service (e.g., Google AI Studio or OpenAI).

