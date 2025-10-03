import streamlit as st
import numpy as np
import io
import soundfile as sf
import tempfile
import os
import time
from typing import Tuple, Optional, Dict, Any
import requests 
import json 
import librosa
import random # Needed for quiz-specific logic

# New dependency for live microphone recording
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False

# --- Gemini API Configuration ---
# In this environment, the API_KEY is left blank and automatically provided at runtime.
API_KEY = "AIzaSyBPW-iSeuA2Ze0hv8268PmUmmd4gaJiJaU" 
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# --- Configuration ---
# Set the page configuration for a professional look
st.set_page_config(
    page_title="Lecture Voice-to-Notes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use a smaller model for faster performance in a web application context
DEFAULT_MODEL = "base" 
SAMPLE_RATE = 16000 # Required sample rate for Whisper

# --- 1. Model Loading (Cached for performance) ---

@st.cache_resource
def load_whisper_model(model_size):
    """Loads the Whisper model and caches it."""
    try:
        from faster_whisper import WhisperModel
        st.info(f"Loading Faster-Whisper model: **{model_size}** (Please wait, this happens once).")
        # Ensure model is loaded with appropriate device settings
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        st.success(f"Model **{model_size}** loaded successfully!")
        return model
    except ImportError:
        st.error("The 'faster-whisper' library is not installed. Please run: `pip install faster-whisper` and `pip install librosa`")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- 2. Centralized Transcription Function (Unchanged) ---

def transcribe_audio_bytes(model, audio_bytes: bytes) -> Optional[Tuple[str, float, str]]:
    """
    Processes audio bytes (from upload or recording), transcribes, and cleans up.
    Returns (transcript, duration, language) or None on failure.
    """
    
    start_time = time.time()
    
    # Create a temporary file to save the audio, as faster-whisper needs a file path
    tmpfile_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
            
            # Use soundfile to read audio data from bytes
            try:
                # Use io.BytesIO to treat the byte array as a file-like object
                audio_data, current_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            except Exception as e:
                st.error(f"Error: Could not read audio data. Invalid format or data structure. Details: {e}")
                return None

            # Resample the audio if the sample rate is not 16000 Hz (required by Whisper)
            if current_sr != SAMPLE_RATE:
                st.warning(f"Resampling audio from {current_sr}Hz to {SAMPLE_RATE}Hz...")
                try:
                    # If the audio is multi-channel, we must handle that before resampling
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    audio_data = librosa.resample(audio_data, orig_sr=current_sr, target_sr=SAMPLE_RATE)
                except ImportError:
                    st.error("The 'librosa' library is required for resampling. Please run: `pip install librosa`")
                    return None
            
            # Ensure it's mono (single channel) if librosa wasn't used
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Write the processed audio data to the temporary WAV file
            sf.write(tmpfile_path, audio_data, SAMPLE_RATE)

        
        # Transcribe using the temporary file path
        segments, info = model.transcribe(tmpfile_path, beam_size=5)
        
        full_transcript = " ".join(segment.text for segment in segments)
        end_time = time.time()
        
        return full_transcript.strip(), end_time - start_time, info.language

    except Exception as e:
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None
    finally:
        # Cleanup: delete the temporary file if it was created
        if tmpfile_path and os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)

# --- 3. Gemini API Function for Summarization (Unchanged) ---

def summarize_text(transcript: str, notes_style: str) -> Optional[str]:
    """
    Calls the Gemini API to summarize the transcript based on the selected style.
    Implements exponential backoff for reliability.
    """
    
    # System Instruction for Note-Taking
    system_prompt_base = "You are a world-class AI note-taker. Your task is to process the following lecture transcript and generate structured, concise notes."
    
    # Adjust prompt based on user's selected style
    if notes_style == "Key Points/Bullet List":
        system_prompt = f"{system_prompt_base} Output the notes as a clear, hierarchical bullet list of key concepts and important facts. Use markdown headings for topics."
    elif notes_style == "Detailed Paragraph Summary":
        system_prompt = f"{system_prompt_base} Output a detailed, dense paragraph summary that covers all major themes and arguments in the lecture."
    else: # Cornell Notes
        system_prompt = f"{system_prompt_base} Format the notes strictly using the Cornell method (Cue Column, Notes Section, Summary at the bottom). Use markdown for headings and lists to structure the output clearly."
        
    user_query = f"Lecture Transcript: {transcript}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        max_retries = 5
        delay = 2  # seconds
        
        # Simple string check for API key presence (based on environment rules)
        if not API_KEY and not os.environ.get('GEMINI_API_KEY'):
            st.error("Gemini API key is required for summarization. Please ensure the key is configured.")
            return None

        for i in range(max_retries):
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                return text
            elif response.status_code == 429: # Rate limit
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None

        st.error("API failed after multiple retries due to rate limiting or persistent error.")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error during API call: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during summarization: {e}")
        return None
    # --- 4. Gemini API Function for Quiz Generation ---

def generate_quiz(transcript: str, num_questions: int) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API to generate a quiz based on the transcript.
    Returns a dictionary with quiz data including questions, options, and correct answers.
    """
    
    system_prompt = f"""You are an expert educator creating multiple-choice quiz questions from lecture content.

Generate exactly {num_questions} multiple-choice questions based on the provided lecture transcript.

CRITICAL INSTRUCTIONS:
1. Each question must have exactly 4 answer options (A, B, C, D)
2. Only ONE option should be correct
3. Questions should test understanding, not just memorization
4. Cover different topics from the lecture
5. Make distractors (wrong answers) plausible but clearly incorrect

OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:
{{
  "quiz_title": "Brief descriptive title for the quiz",
  "questions": [
    {{
      "question_text": "The question text here?",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_answer": "The exact text of the correct option"
    }}
  ]
}}

DO NOT include any text outside the JSON structure. DO NOT use markdown code blocks or backticks.
Your entire response must be a single valid JSON object."""

    user_query = f"Lecture Transcript:\n\n{transcript}\n\nGenerate {num_questions} quiz questions."

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        max_retries = 5
        delay = 2
        
        if not API_KEY and not os.environ.get('GEMINI_API_KEY'):
            st.error("Gemini API key is required for quiz generation.")
            return None

        for i in range(max_retries):
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                
                # Clean up the response - remove markdown code blocks if present
                text = text.replace('```json', '').replace('```', '').strip()
                
                # Parse JSON
                try:
                    quiz_data = json.loads(text)
                    
                    # Validate the structure
                    if not isinstance(quiz_data, dict):
                        st.error("Invalid quiz format: Response is not a dictionary")
                        return None
                    
                    if 'questions' not in quiz_data:
                        st.error("Invalid quiz format: Missing 'questions' field")
                        return None
                    
                    questions = quiz_data['questions']
                    
                    if not isinstance(questions, list) or len(questions) == 0:
                        st.error("Invalid quiz format: 'questions' must be a non-empty list")
                        return None
                    
                    # Validate each question
                    for idx, q in enumerate(questions):
                        if not isinstance(q, dict):
                            st.error(f"Question {idx+1} is not a dictionary")
                            return None
                        
                        required_keys = ['question_text', 'options', 'correct_answer']
                        for key in required_keys:
                            if key not in q:
                                st.error(f"Question {idx+1} missing required field: {key}")
                                return None
                        
                        if not isinstance(q['options'], list) or len(q['options']) < 2:
                            st.error(f"Question {idx+1} must have at least 2 options")
                            return None
                        
                        if q['correct_answer'] not in q['options']:
                            st.error(f"Question {idx+1}: correct_answer must be one of the options")
                            return None
                    
                    return quiz_data
                    
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse quiz JSON: {e}")
                    st.error(f"Raw response: {text[:500]}...")
                    return None
                    
            elif response.status_code == 429:
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None

        st.error("API failed after multiple retries.")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error during quiz generation: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during quiz generation: {e}")
        return None
# --- 5. Streamlit Application Layout Functions ---

def notes_generation_tab():
    """Handles the UI and logic for generating summarized notes."""
    
    if 'full_transcript' not in st.session_state or not st.session_state['full_transcript']:
        st.warning("Please upload and transcribe an audio file first in the 'Upload Audio File' tab.")
        return

    transcript = st.session_state['full_transcript']
    
    st.markdown("---")
    st.subheader("🧠 2. Generate Summarized Notes (Powered by Gemini)")
    
    # Input for Note Style
    notes_style = st.selectbox(
        "Select Note Style:",
        ["Cornell Notes (Cues/Notes/Summary)", "Key Points/Bullet List", "Detailed Paragraph Summary"],
        key="note_style_select"
    )

    if st.button("✨ Generate Notes", type="primary", key="summarize_btn"):
        with st.spinner(f"Generating notes in {notes_style} format..."):
            
            # Call the summarization function
            summary = summarize_text(transcript, notes_style)
            
            if summary:
                st.session_state['summarized_notes'] = summary
                st.session_state['summary_style'] = notes_style
            else:
                # If summary fails, clear state to prevent displaying old result
                st.session_state['summarized_notes'] = "" 
                
        # Rerun to display summary if successful
        if 'summarized_notes' in st.session_state and st.session_state['summarized_notes']:
            st.rerun()

    # Display Summarized Notes
    if 'summarized_notes' in st.session_state and st.session_state['summarized_notes']:
        st.markdown("---")
        st.subheader(f"✅ Summarized Notes: {st.session_state['summary_style']}")
        
        # Use markdown to render the structured notes
        st.markdown(st.session_state['summarized_notes'])
        
        st.download_button(
            label="Download Summarized Notes (TXT)",
            data=st.session_state['summarized_notes'],
            file_name="summarized_lecture_notes.txt",
            mime="text/plain"
        )
        
    st.markdown("---")
    st.caption("Review the raw transcript used for note generation:")
    st.text_area("Raw Transcript for Review", transcript, height=150, disabled=True)


def quiz_generation_tab():
    """Handles the UI and logic for generating and taking the quiz."""
    
    if 'full_transcript' not in st.session_state or not st.session_state['full_transcript']:
        st.warning("Please upload and transcribe an audio file first in the 'Upload Audio File' tab.")
        return

    transcript = st.session_state['full_transcript']
    
    st.markdown("---")
    st.subheader("📝 3. Generate and Take a Quiz (Powered by Gemini)")

    col1, col2 = st.columns([1, 2])
    
    # Configuration
    num_questions = col1.slider(
        "Number of Questions:",
        min_value=3, max_value=10, value=5, step=1, key="num_questions_slider"
    )
    
    # Generate Button
    if col2.button("🚀 Generate New Quiz", type="primary", key="generate_quiz_btn"):
        with st.spinner(f"Generating a {num_questions}-question quiz..."):
            quiz_data = generate_quiz(transcript, num_questions)
            
            if quiz_data:
                # Initialize score and answers
                st.session_state['quiz_data'] = quiz_data
                st.session_state['quiz_answers'] = {str(i): None for i in range(num_questions)}
                st.session_state['quiz_submitted'] = False
                st.session_state['quiz_score'] = 0
            else:
                st.session_state['quiz_data'] = None
                
        if st.session_state.get('quiz_data'):
            st.rerun()

    st.markdown("---")
    
    # Quiz Display and Logic
    if 'quiz_data' in st.session_state and st.session_state['quiz_data']:
        quiz_data = st.session_state['quiz_data']
        questions = quiz_data['questions']
        
        st.info(f"Quiz Title: **{quiz_data.get('quiz_title', 'Lecture Review Quiz')}**")

        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['question_text']}")
                
                # Create a list of all options
                options_with_correct = q['options']
                
                # Shuffle the options only when the quiz is first loaded
                if f'options_shuffled_{i}' not in st.session_state:
                     # This runs only once per quiz generation
                    random.shuffle(options_with_correct)
                    st.session_state[f'options_shuffled_{i}'] = options_with_correct

                shuffled_options = st.session_state[f'options_shuffled_{i}']

                # Use a unique key for each radio group
                key = f"q_{i}_answer"
                
                # Pre-select the user's previous answer if available
                default_index = 0
                if key in st.session_state and st.session_state[key] is not None:
                     try:
                         default_index = shuffled_options.index(st.session_state[key])
                     except ValueError:
                         # Handle case where previous answer isn't in options (shouldn't happen with proper state management)
                         pass

                # Store the user's answer in a temporary variable
                answer = st.radio(
                    "Select an answer:",
                    shuffled_options,
                    index=default_index,
                    key=key,
                    disabled=st.session_state.get('quiz_submitted', False) # Disable after submission
                )
                user_answers[str(i)] = answer

                # Display feedback if submitted
                if st.session_state.get('quiz_submitted', False):
                    correct = (answer == q['correct_answer'])
                    if correct:
                        st.success(f"Correct! The answer is: **{q['correct_answer']}**")
                    else:
                        st.error(f"Incorrect. Your answer: **{answer}**. The correct answer is: **{q['correct_answer']}**.")
                
                st.markdown("---")
            
            submit_button = st.form_submit_button(
                "Submit Quiz and See Score", 
                disabled=st.session_state.get('quiz_submitted', False)
            )

            # Store answers on every form submission (even without clicking the button)
            # This is key for state persistence when the radio button is clicked
            for i in range(len(questions)):
                st.session_state['quiz_answers'][str(i)] = st.session_state[f"q_{i}_answer"]
            
            # Handle submission logic when the button is explicitly pressed
            if submit_button:
                score = 0
                for i, q in enumerate(questions):
                    if st.session_state['quiz_answers'].get(str(i)) == q['correct_answer']:
                        score += 1

                st.session_state['quiz_score'] = score
                st.session_state['quiz_submitted'] = True
                
                # Rerun to display the feedback (disabled/score)
                st.rerun()

        # Display final score outside the form after submission
        if st.session_state.get('quiz_submitted', False):
            score = st.session_state['quiz_score']
            total = len(questions)
            st.balloons()
            st.metric("Final Score", f"{score} / {total}", f"{round((score/total)*100)}%")
            if score == total:
                 st.success("Perfect score! You mastered the lecture material! 🎓")
            elif score > total * 0.7:
                 st.info("Great job! You have a strong grasp of the material. 💪")
            else:
                 st.warning("Review the notes for the questions you missed. Keep studying! 📖")
            
            # Button to clear submission state and start a new attempt
            if st.button("Start New Attempt/Clear Feedback", key="new_attempt_btn"):
                 st.session_state['quiz_submitted'] = False
                 # Clear option shuffling keys to allow options to be reshuffled if desired
                 for i in range(len(questions)):
                    if f'options_shuffled_{i}' in st.session_state:
                         del st.session_state[f'options_shuffled_{i}']
                 st.rerun()
    else:
        st.info("Generate a quiz to begin testing your knowledge!")


def main():
    
    # Initialize session state for transcript persistence and tab management
    if 'full_transcript' not in st.session_state:
        st.session_state['full_transcript'] = ""
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = 'Upload Audio File'
    if 'summarized_notes' not in st.session_state:
        st.session_state['summarized_notes'] = ""
    # NEW: Initialize quiz state
    if 'quiz_data' not in st.session_state:
        st.session_state['quiz_data'] = None
    if 'quiz_submitted' not in st.session_state:
        st.session_state['quiz_submitted'] = False
        

    st.title("🗣️ Lecture Voice-to-Notes & Quiz Generator")
    st.markdown("Convert long audio recordings or files into detailed lecture notes and test your retention.")
    
    st.markdown("---")

    # Sidebar for Model Selection and API Key Input (Unchanged)
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox(
            "Select Transcription Model Size:",
            ["base", "tiny", "medium", "large-v3"],
            index=0 # Default to 'base'
        )
        st.markdown("""
        * **Base/Tiny:** Fastest, lowest accuracy.
        * **Medium:** Good balance of speed and accuracy.
        * **Large-v3:** Slowest, highest accuracy.
        """)
        
        st.header("Generative API")
        
        # Display API status (checks against the defined API_KEY variable)
        if API_KEY == "":
            st.error("⚠️ Gemini API Key status: Missing")
        else:
            st.success("✅ Gemini API Key status: Ready")

        
    # Load the Whisper model based on selection
    model = load_whisper_model(model_choice)
    
    if model is None:
        return # Stop if model loading failed

    # --- Main Content: Tabbed Interface ---
    # NEW TAB ADDED: "Quiz Generation"
    tab_titles = ["Upload Audio File", "Notes Generation", "Quiz Generation", "Record Live Audio"]
    
    # Determine which tab is active (default is upload, switch to notes after transcription)
    tabs = st.tabs(tab_titles)
    
    # --- TAB 1: Upload File (Transcription) (Logic Unchanged, now in tabs[0]) ---
    with tabs[0]:
        st.subheader("1. Upload Lecture Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, M4A, FLAC, OGG, etc.)", 
            type=["mp3", "wav", "m4a", "flac", "ogg"]
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format=uploaded_file.type, start_time=0)
            
            if st.button("▶️ Transcribe Uploaded Lecture", type="primary", key="upload_btn"):
                
                # Show a spinner while processing
                with st.spinner(f"Transcribing {uploaded_file.name}..."):
                    
                    audio_bytes = uploaded_file.read()
                    
                    result = transcribe_audio_bytes(model, audio_bytes)
                
                if result:
                    transcript, duration, language = result
                    
                    # Store transcription results in session state
                    st.session_state['full_transcript'] = transcript
                    st.session_state['transcript_language'] = language
                    st.session_state['transcript_source'] = "File Upload"
                    st.session_state['transcript_model'] = model_choice
                    st.session_state['transcript_duration'] = duration
                    
                    # Clear summary and quiz to force regeneration and switch tab
                    st.session_state['summarized_notes'] = "" 
                    st.session_state['quiz_data'] = None 
                    st.session_state['current_tab'] = 'Notes Generation'
                    st.toast("Transcription complete! Notes tab is ready.")
                    
                    # Rerun to switch the active tab visually
                    st.rerun() 

    # --- TAB 2: Notes Generation (Logic Unchanged, now in tabs[1]) ---
    with tabs[1]:
        notes_generation_tab()
        
        # Display current transcription details if available
        if st.session_state['full_transcript']:
            st.markdown("---")
            st.subheader("Current Transcription Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Source", st.session_state.get('transcript_source', 'N/A'))
            col2.metric("Detected Language", st.session_state.get('transcript_language', 'N/A').upper())
            col3.metric("Transcription Time", f"{st.session_state.get('transcript_duration', 0):.2f} s")
            st.caption(f"Using Whisper model: {st.session_state.get('transcript_model', 'N/A')}")
            
            st.download_button(
                label="Download Raw Transcript (TXT)",
                data=st.session_state['full_transcript'],
                file_name="raw_lecture_transcript.txt",
                mime="text/plain"
            )

    # --- NEW TAB 3: Quiz Generation ---
    with tabs[2]:
        quiz_generation_tab()

    # --- TAB 4: Record Live Audio (Logic Unchanged, now in tabs[3]) ---
    with tabs[3]:
        st.subheader("4. Record Live Audio Lecture")
        
        if not HAS_MIC_RECORDER:
            st.error(
                "The required library for microphone access, `streamlit-mic-recorder`, "
                "is not installed. Please install it using the command in the chat."
            )
            st.markdown(
                "**Recommended Alternative:** Continue using the **Upload Audio File** tab."
            )
            return

        st.info("Recording requires an up-to-date Chrome or Firefox browser and permission to access your microphone.")
        
        # Capture audio data as a WAV file byte string
        audio_data = mic_recorder(
            start_prompt="Click to Start Recording",
            stop_prompt="Click to Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key='mic_recorder_key'
        )

        # Check if the recording was completed and contains bytes
        if audio_data and isinstance(audio_data, dict) and audio_data.get('bytes'):
            st.success("Audio captured successfully! Transcribing...")
            
            with st.spinner("Transcribing live audio..."):
                audio_bytes = audio_data['bytes']
                
                # Directly transcribe the captured bytes
                result = transcribe_audio_bytes(model, audio_bytes)
            
            if result:
                transcript, duration, language = result
                
                # Store transcription results in session state
                st.session_state['full_transcript'] = transcript
                st.session_state['transcript_language'] = language
                st.session_state['transcript_source'] = "Live Recording"
                st.session_state['transcript_model'] = model_choice
                st.session_state['transcript_duration'] = duration
                
                # Clear summary and quiz and switch tab to view notes
                st.session_state['summarized_notes'] = ""
                st.session_state['quiz_data'] = None
                st.session_state['current_tab'] = 'Notes Generation'
                st.toast("Live recording transcribed! Notes tab is ready.")
                st.rerun() # Rerun to switch tabs and display results
        elif audio_data is not None:
             st.warning("Recording was stopped or failed to capture audio data. Please try again.")

if __name__ == "__main__":
    main()