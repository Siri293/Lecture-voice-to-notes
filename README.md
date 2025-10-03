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

## 🛠️ Requirements & Prerequisites

### Technical Requirements
* **Python:** 3.8 or higher.
* **API Key:** A **Gemini API Key** is mandatory for all summarization and quiz generation features.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd lecture-voice-to-notes
    ```

2.  **Install Libraries:** Run the following command to install all necessary Python packages:
    ```bash
    pip install streamlit numpy soundfile librosa requests faster-whisper streamlit-mic-recorder
    ```
    *Note: `faster-whisper` relies on `ctranslate2`, and `librosa` may require system dependencies on some operating systems. Consult their documentation if you encounter specific build errors.*

---

## 🚀 Setup and Running the App

### 1. Set Your API Key

The application relies on the Gemini API for its core intelligence features. You must set your API key as an environment variable.

| Variable Name | Purpose |
| :--- | :--- |
| `GEMINI_API_KEY` | Your key starting with `AIza...` |

**Example (Linux/macOS):**
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
