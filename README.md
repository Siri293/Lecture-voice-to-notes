# Lecture-voice-to-notes# 🗣️ Lecture Voice-to-Notes & Quiz Generator

A full-stack study tool built with Streamlit, Faster Whisper, and the Gemini API. This application converts long lecture audio into organized, customizable study notes and generates tailored multiple-choice quizzes to test your understanding.

---

## ✨ Features

This application provides a seamless pipeline from audio input to effective knowledge retention:

### 1. Robust Transcription (Powered by Faster Whisper)
* **Flexible Input:** Supports uploading common audio formats (`.mp3`, `.wav`, `.m4a`, etc.) and **live microphone recording** using `streamlit-mic-recorder`.
* **Model Options:** Allows selection of various Whisper models (`tiny`, `base`, `medium`, `large-v3`) to prioritize speed or accuracy.
* **Audio Pre-processing:** Automatically handles necessary resampling (`librosa`) and format conversion for reliable transcription.

### 2. Intelligent Note Synthesis (Powered by Gemini)
* **Structured Notes:** Generates concise and structured notes from the raw transcript.
* **Customizable Styles:** Choose from professional academic formats:
    * **Cornell Notes** (Cue Column, Notes Section, Summary)
    * **Key Points/Bullet List** (Hierarchical structure)
    * **Detailed Paragraph Summary**
* **Downloadable Output:** Both the raw transcript and the summarized notes can be downloaded as text files.

### 3. Knowledge Assessment (Powered by Gemini)
* **Quiz Generation:** Creates multiple-choice questions (3-10) directly based on the lecture content.
* **Interactive Testing:** Features a live quiz interface with submission handling and immediate correctness feedback.
* **Scoring & Feedback:** Provides a final score, percentage, and personalized study tips.

---
Requirements:
streamlit
numpy
soundfile
faster-whisper
librosa
requests
streamlit-mic-recorder

🚀 Installation Steps
Step 1: Clone the Repository (If applicable)
If your code is hosted on GitHub, start by cloning the repository.

Bash

git clone [YOUR_REPO_URL]
cd lecture-voice-to-notes
Step 2: Install Libraries
This application relies on several specialized libraries for audio processing, transcription, and the user interface.

Run the following command in your terminal to install all necessary packages:

Bash

pip install streamlit numpy soundfile librosa requests faster-whisper streamlit-mic-recorder
Brief explanation of the key libraries:

faster-whisper: Used for fast and efficient transcription of the audio.

librosa: Used for resampling and processing complex audio formats to meet Whisper's requirements (16000 Hz, mono).

streamlit-mic-recorder: The custom component that enables live microphone input in the Streamlit app.

Step 3: Set the Gemini API Key
The application relies on this key to access the Gemini model for summarization and quiz generation.

Set the key as an environment variable in your terminal session before running the app.

On Linux/macOS:

Bash

export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
On Windows (Command Prompt):

Bash

set GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
Note: The provided script has a placeholder API_KEY variable ("AIzaSyBPW-iSeuA2Ze0hv8268PmUmmd4gaJiJaU"), which will likely fail unless it is a valid, configured key. It is best practice to use the environment variable as shown above, or Streamlit Secrets.

▶️ Running the Application
Once all prerequisites are met, run the Streamlit application from your terminal:

Bash

streamlit run your_app_script_name.py
(Replace your_app_script_name.py with the actual file name of your Python script.)

The app will launch in your web browser at http://localhost:8501.
