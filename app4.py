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
import random 
import hashlib # For securely hashing passwords

# New dependency for live microphone recording
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False
    
# --- Security Warning and Gemini API Configuration ---
# WARNING: Replace with your actual key or use st.secrets!
API_KEY = "AIzaSyBPW-iSeuA2Ze0hv8268PmUmmd4gaJiJaU" 
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# --- Configuration ---
st.set_page_config(
    page_title="Lecture Voice-to-Notes",
    layout="wide",
    initial_sidebar_state="expanded")
    
DEFAULT_MODEL = "base" 
SAMPLE_RATE = 16000

# --- Helper Functions for Authentication ---

def hash_password(password: str) -> str:
    """Hashes the password using SHA256 for simple session-based storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_account(username, password):
    """Simulates creating a user account and stores it in session state."""
    if not username or not password:
        st.error("Username and password cannot be empty.")
        return False
        
    if username in st.session_state['user_db']:
        st.error("Username already exists. Please log in or choose another name.")
        return False
        
    hashed_password = hash_password(password)
    # Store the user's data (hashed password and an empty data dictionary)
    st.session_state['user_db'][username] = {
        'password': hashed_password,
        'data': {} # Future placeholder for saving notes/quizzes
    }
    st.session_state['is_authenticated'] = True
    st.session_state['current_user'] = username
    st.success(f"Account '{username}' created successfully! You are now logged in.")
    st.rerun()

def user_login(username, password):
    """Simulates user login against stored session state accounts."""
    if username not in st.session_state['user_db']:
        st.error("Login failed: User not found.")
        return False
        
    hashed_input = hash_password(password)
    stored_hash = st.session_state['user_db'][username]['password']
    
    if hashed_input == stored_hash:
        st.session_state['is_authenticated'] = True
        st.session_state['current_user'] = username
        st.success(f"Welcome back, {username}!")
        st.rerun()
    else:
        st.error("Login failed: Incorrect password.")
        return False

def logout():
    """Logs the user out and clears sensitive session data."""
    st.session_state['is_authenticated'] = False
    st.session_state['current_user'] = None
    st.session_state['full_transcript'] = "" # Clear transcript on logout
    st.session_state['summarized_notes'] = ""
    st.session_state['quiz_data'] = None
    st.session_state['flashcard_data'] = None
    st.sidebar.info("Logged out successfully.")
    st.rerun()

# --- 1. Model Loading (Unchanged) ---
@st.cache_resource
def load_whisper_model(model_size):
    """Loads the Whisper model and caches it."""
    try:
        from faster_whisper import WhisperModel
        st.info(f"Loading Faster-Whisper model: **{model_size}** (Please wait, this happens once).")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        st.success(f"Model **{model_size}** loaded successfully!")
        return model
    except ImportError:
        st.error("The 'faster-whisper' library is not installed. Please run: `pip install faster-whisper` and `pip install librosa`")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- 2. Transcription Function (Unchanged) ---
def transcribe_audio_bytes(model, audio_bytes: bytes) -> Optional[Tuple[str, float, str]]:
    """
    Processes audio bytes (from upload or recording), transcribes, and cleans up.
    Returns (transcript, duration, language) or None on failure.
    """
    if model is None:
        st.error("Transcription model is not loaded.")
        return None
        
    start_time = time.time()
    tmpfile_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
                        
            try:
                audio_data, current_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            except Exception as e:
                st.error(f"Error: Could not read audio data. Invalid format or data structure. Details: {e}")
                return None
            
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            if current_sr != SAMPLE_RATE:
                # Resampling logic requires librosa
                st.warning(f"Resampling audio from {current_sr}Hz to {SAMPLE_RATE}Hz...")
                try:
                    audio_data = librosa.resample(audio_data, orig_sr=current_sr, target_sr=SAMPLE_RATE)
                except ImportError:
                    st.error("The 'librosa' library is required for resampling. Please run: `pip install librosa`")
                    return None
                        
            sf.write(tmpfile_path, audio_data, SAMPLE_RATE)
                
        segments, info = model.transcribe(tmpfile_path, beam_size=5)
        full_transcript = " ".join(segment.text for segment in segments)
        end_time = time.time()
        
        return full_transcript.strip(), end_time - start_time, info.language
    except Exception as e:
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None
    finally:
        if tmpfile_path and os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)

# --- 3. Summarization Function (Unchanged) ---
def summarize_text(transcript: str, notes_style: str) -> Optional[str]:
    """Calls the Gemini API to summarize the transcript based on the selected style."""
    
    system_prompt_base = "You are a world-class AI note-taker. Your task is to process the following lecture transcript and generate structured, concise notes."
    
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
        delay = 2
        
        if not API_KEY:
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
            elif response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
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

# --- 4. Flashcard Generation Function (Unchanged) ---
def generate_flashcards(transcript: str, num_cards: int) -> Optional[Dict[str, Any]]:
    # ... (function body unchanged) ...
    """Calls the Gemini API to generate flashcards based on the transcript."""
    
    system_prompt = f"""You are an expert educator creating flashcards from lecture content.
Generate exactly {num_cards} flashcards based on the provided lecture transcript.
CRITICAL INSTRUCTIONS:
1. Each flashcard should have a clear question/concept on the front
2. Each flashcard should have a concise, accurate answer on the back
3. Cover the most important concepts from the lecture
4. Make questions clear and focused
5. Keep answers informative but brief (2-4 sentences ideal)
OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:
{{
  "flashcard_set_title": "Brief descriptive title for the flashcard set",
  "flashcards": [
    {{
      "front": "Question or concept to learn",
      "back": "Clear, concise answer or explanation"
    }}
  ]
}}
DO NOT include any text outside the JSON structure. DO NOT use markdown code blocks or backticks.
Your entire response must be a single valid JSON object."""
    user_query = f"Lecture Transcript:\n\n{transcript}\n\nGenerate {num_cards} flashcards."
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        max_retries = 5
        delay = 2
        
        if not API_KEY:
            st.error("Gemini API key is required for flashcard generation.")
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
                text = text.replace('```json', '').replace('```', '').strip() 
                
                try:
                    flashcard_data = json.loads(text)
                    
                    if not isinstance(flashcard_data, dict) or 'flashcards' not in flashcard_data or not isinstance(flashcard_data['flashcards'], list) or len(flashcard_data['flashcards']) == 0:
                        st.error("Invalid flashcard format: Check LLM output structure.")
                        return None
                        
                    return flashcard_data
                        
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse flashcard JSON: {e}")
                    st.error(f"Raw response: {text[:500]}...")
                    return None
                    
            elif response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None
        st.error("API failed after multiple retries.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during flashcard generation: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during flashcard generation: {e}")
        return None

# --- 5. Quiz Generation Function (Unchanged) ---
def generate_quiz(transcript: str, num_questions: int) -> Optional[Dict[str, Any]]:
    # ... (function body unchanged) ...
    """Calls the Gemini API to generate a quiz based on the transcript."""
    
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
        
        if not API_KEY:
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
                text = text.replace('```json', '').replace('```', '').strip()
                
                try:
                    quiz_data = json.loads(text)
                    
                    if not isinstance(quiz_data, dict) or 'questions' not in quiz_data or not isinstance(quiz_data['questions'], list) or len(quiz_data['questions']) == 0:
                        st.error("Invalid quiz format: Check LLM output structure.")
                        return None
                        
                    return quiz_data
                        
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse quiz JSON: {e}")
                    st.error(f"Raw response: {text[:500]}...")
                    return None
                    
            elif response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
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

# --- 6. Tab Functions (Unchanged, but now hidden if not logged in) ---

def notes_generation_tab():
    if not st.session_state.get('is_authenticated'):
        st.info("Log in or create an account to access the notes generator.")
        return
        
    if 'full_transcript' not in st.session_state or not st.session_state['full_transcript']:
        st.warning("Please upload and transcribe an audio file first in the 'Upload Audio File' tab.")
        return
        
    transcript = st.session_state['full_transcript']
    
    st.markdown("---")
    st.subheader("🧠 2. Generate Summarized Notes (Powered by Gemini)")
    
    # ... (rest of the notes_generation_tab logic remains the same) ...
    notes_style = st.selectbox(
        "Select Note Style:",
        ["Cornell Notes (Cues/Notes/Summary)", "Key Points/Bullet List", "Detailed Paragraph Summary"],
        key="note_style_select"
    )
    
    if st.button("✨ Generate Notes", type="primary", key="summarize_btn"):
        with st.spinner(f"Generating notes in {notes_style} format..."):
            summary = summarize_text(transcript, notes_style)
            
            if summary:
                st.session_state['summarized_notes'] = summary
                st.session_state['summary_style'] = notes_style
            else:
                st.session_state['summarized_notes'] = ""
            
            if 'summarized_notes' in st.session_state and st.session_state['summarized_notes']:
                st.rerun()

    if 'summarized_notes' in st.session_state and st.session_state['summarized_notes']:
        st.markdown("---")
        st.subheader(f"✅ Summarized Notes: {st.session_state['summary_style']}")
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
    if not st.session_state.get('is_authenticated'):
        st.info("Log in or create an account to access the quiz generator.")
        return
        
    if 'full_transcript' not in st.session_state or not st.session_state['full_transcript']:
        st.warning("Please upload and transcribe an audio file first in the 'Upload Audio File' tab.")
        return
        
    transcript = st.session_state['full_transcript']
    
    st.markdown("---")
    st.subheader("📝 3. Generate and Take a Quiz (Powered by Gemini)")
    # ... (rest of the quiz_generation_tab logic remains the same) ...
    col1, col2 = st.columns([1, 2])
    
    num_questions = col1.slider(
        "Number of Questions:",
        min_value=3, max_value=10, value=5, step=1, key="num_questions_slider"
    )
    
    if col2.button("🚀 Generate New Quiz", type="primary", key="generate_quiz_btn"):
        with st.spinner(f"Generating a {num_questions}-question quiz..."):
            quiz_data = generate_quiz(transcript, num_questions)
            
            if quiz_data:
                st.session_state['quiz_data'] = quiz_data
                st.session_state['quiz_answers'] = {str(i): None for i in range(num_questions)}
                st.session_state['quiz_submitted'] = False
                st.session_state['quiz_score'] = 0
                
                for i in range(num_questions):
                    if f'options_shuffled_{i}' in st.session_state: 
                        del st.session_state[f'options_shuffled_{i}']
                        
            else:
                st.session_state['quiz_data'] = None
            
            if st.session_state.get('quiz_data'):
                st.rerun()

    st.markdown("---")
    
    if 'quiz_data' in st.session_state and st.session_state['quiz_data']:
        quiz_data = st.session_state['quiz_data']
        questions = quiz_data['questions']
        
        st.info(f"Quiz Title: **{quiz_data.get('quiz_title', 'Lecture Review Quiz')}**")
        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['question_text']}")
                
                options_with_correct = q['options']
                
                if f'options_shuffled_{i}' not in st.session_state:
                    shuffled_options = list(q['options']) 
                    random.shuffle(shuffled_options)
                    st.session_state[f'options_shuffled_{i}'] = shuffled_options
                shuffled_options = st.session_state[f'options_shuffled_{i}']
                
                key = f"q_{i}_answer"
                
                default_index = 0
                if key in st.session_state and st.session_state[key] is not None: 
                     try: 
                         default_index = shuffled_options.index(st.session_state[key]) 
                     except ValueError: 
                         pass
                         
                answer = st.radio(
                    "Select an answer:",
                    shuffled_options,
                    index=default_index,
                    key=key,
                    disabled=st.session_state.get('quiz_submitted', False)
                )
                user_answers[str(i)] = answer
                
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
            
            for i in range(len(questions)):
                st.session_state['quiz_answers'][str(i)] = st.session_state[f"q_{i}_answer"]
                        
            if submit_button:
                score = 0
                for i, q in enumerate(questions):
                    if st.session_state['quiz_answers'].get(str(i)) == q['correct_answer']:
                        score += 1
                st.session_state['quiz_score'] = score
                st.session_state['quiz_submitted'] = True
                st.rerun()

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
                        
            if st.button("Start New Attempt/Clear Feedback", key="new_attempt_btn"):
                 st.session_state['quiz_submitted'] = False
                 for i in range(len(questions)):
                    if f'options_shuffled_{i}' in st.session_state: 
                         del st.session_state[f'options_shuffled_{i}']
                 st.rerun()
                 
    else:
        st.info("Generate a quiz to begin testing your knowledge!")

def flashcard_study_tab():
    if not st.session_state.get('is_authenticated'):
        st.info("Log in or create an account to access the flashcard study tool.")
        return
        
    if 'full_transcript' not in st.session_state or not st.session_state['full_transcript']:
        st.warning("Please upload and transcribe an audio file first in the 'Upload Audio File' tab.")
        return
        
    transcript = st.session_state['full_transcript']
    
    st.markdown("---")
    st.subheader("🗂️ 4. Generate and Study Flashcards (Powered by Gemini)")
    # ... (rest of the flashcard_study_tab logic remains the same) ...
    col1, col2 = st.columns([1, 2])
    
    num_cards = col1.slider(
        "Number of Flashcards:",
        min_value=5, max_value=20, value=10, step=1, key="num_cards_slider"
    )
    
    if col2.button("🎴 Generate New Flashcards", type="primary", key="generate_flashcards_btn"):
        with st.spinner(f"Generating {num_cards} flashcards..."):
            flashcard_data = generate_flashcards(transcript, num_cards)
            
            if flashcard_data:
                st.session_state['flashcard_data'] = flashcard_data
                st.session_state['current_card_index'] = 0
                st.session_state['card_flipped'] = False
                st.session_state['cards_marked'] = {}
            else:
                st.session_state['flashcard_data'] = None
            
            if st.session_state.get('flashcard_data'):
                st.rerun()

    st.markdown("---")
    
    if 'flashcard_data' in st.session_state and st.session_state['flashcard_data']:
        flashcard_data = st.session_state['flashcard_data']
        flashcards = flashcard_data['flashcards']
        
        st.info(f"📚 **{flashcard_data.get('flashcard_set_title', 'Lecture Flashcards')}**")
        
        if 'current_card_index' not in st.session_state:
            st.session_state['current_card_index'] = 0
        if 'card_flipped' not in st.session_state:
            st.session_state['card_flipped'] = False
        if 'cards_marked' not in st.session_state:
            st.session_state['cards_marked'] = {}
            
        current_index = st.session_state['current_card_index']
        current_card = flashcards[current_index]
        is_flipped = st.session_state['card_flipped']
        
        progress = (current_index + 1) / len(flashcards)
        st.progress(progress, text=f"Card {current_index + 1} of {len(flashcards)}")
        
        st.markdown("### 🎴 Flashcard")
        
        card_placeholder = st.empty()
        
        if not is_flipped:
            card_placeholder.info(f"**QUESTION:**\n\n## {current_card['front']}")
        else:
            card_placeholder.success(f"**ANSWER:**\n\n{current_card['back']}")
            
        st.markdown("")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(current_index == 0), key="prev_card"):
                st.session_state['current_card_index'] -= 1
                st.session_state['card_flipped'] = False
                st.rerun()
                
        with col2:
            flip_label = "Show Answer" if not is_flipped else "Show Question"
            if st.button(f"🔄 {flip_label}", key="flip_card"):
                st.session_state['card_flipped'] = not st.session_state['card_flipped']
                st.rerun()
                
        with col3:
            if st.button("➡️ Next", disabled=(current_index == len(flashcards) - 1), key="next_card"):
                st.session_state['current_card_index'] += 1
                st.session_state['card_flipped'] = False
                st.rerun()
                
        with col4:
            current_status = st.session_state['cards_marked'].get(current_index, "unmarked")
            if current_status == "unmarked":
                if st.button("⭐ Mark Mastered", key="mark_mastered"):
                    st.session_state['cards_marked'][current_index] = "mastered"
                    st.rerun()
            elif current_status == "mastered":
                if st.button("❌ Unmark", key="unmark_card"):
                    st.session_state['cards_marked'][current_index] = "unmarked"
                    st.rerun()
                    
        st.markdown("---")
        st.subheader("📊 Study Progress")
        
        mastered_count = sum(1 for status in st.session_state['cards_marked'].values() if status == "mastered")
        total_cards = len(flashcards)
        mastery_percentage = (mastered_count / total_cards) * 100 if total_cards > 0 else 0
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Total Cards", total_cards)
        stat_col2.metric("Mastered", mastered_count)
        stat_col3.metric("Mastery %", f"{mastery_percentage:.1f}%")
        
        with st.expander("📋 View All Flashcards"):
            for idx, card in enumerate(flashcards):
                status_emoji = "✅" if st.session_state['cards_marked'].get(idx) == "mastered" else "⬜"
                st.markdown(f"**{status_emoji} Card {idx + 1}:** {card['front']}")
                st.markdown(f"*Answer:* {card['back']}")
                st.markdown("---")
                
        if st.button("🔄 Reset Study Progress", key="reset_flashcards"):
            st.session_state['current_card_index'] = 0
            st.session_state['card_flipped'] = False
            st.session_state['cards_marked'] = {}
            st.rerun()
            
    else:
        st.info("Generate flashcards to begin studying the lecture material!")

# --- 7. Audio Upload Tab Function (Hidden if not logged in) ---
def audio_upload_tab(model_choice):
    if not st.session_state.get('is_authenticated'):
        st.info("Log in or create an account to start transcribing audio.")
        return
        
    st.markdown("---")
    st.subheader("🎤 1. Upload or Record Lecture Audio")
    
    # ... (rest of the audio_upload_tab logic remains the same) ...
    audio_source = st.radio(
        "Select Audio Source:", 
        ["Upload File", "Record Live (Browser)"],
        key="audio_source_select"
    )
    
    audio_bytes = None
    uploaded_file_name = "Live Recording"
    
    if audio_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Audio File (.mp3, .wav, .ogg, etc.)", 
            type=['mp3', 'wav', 'ogg', 'flac'],
            key="audio_uploader"
        )
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            uploaded_file_name = uploaded_file.name
            st.audio(audio_bytes, format=uploaded_file.type)
            
    elif audio_source == "Record Live (Browser)":
        if HAS_MIC_RECORDER:
            st.info("Start recording your lecture. The recording stops automatically when you hit 'Stop'.")
            audio_data = mic_recorder(
                start_prompt="Start Recording", 
                stop_prompt="Stop Recording", 
                key='mic_recorder',
                just_once=False
            )
            if audio_data and audio_data['bytes']:
                audio_bytes = audio_data['bytes']
                st.audio(audio_bytes, format='audio/wav')
                uploaded_file_name = "Live Recording"
        else:
            st.error("Live recording library not installed. Please run: `pip install streamlit-mic-recorder`")


    st.markdown("---")
    
    if audio_bytes is not None:
        if st.button("▶️ Transcribe Audio", type="primary", key="transcribe_btn"):
            model = load_whisper_model(model_choice)
            if model:
                with st.spinner(f"Transcribing '{uploaded_file_name}' using {model_choice} model..."):
                    transcript_data = transcribe_audio_bytes(model, audio_bytes)
                    
                    if transcript_data:
                        transcript, duration, language = transcript_data
                        st.session_state['full_transcript'] = transcript
                        st.session_state['transcription_file_name'] = uploaded_file_name
                        st.session_state['transcription_duration'] = duration
                        st.session_state['transcription_language'] = language
                        st.session_state['current_tab'] = 'Summarized Notes' 
                        st.success(f"Transcription complete in {duration:.2f}s (Language: {language.upper()})!")
                        st.rerun()
                    else:
                        st.error("Transcription failed.")
    else:
        st.info("Waiting for audio file upload or live recording...")


    if st.session_state['full_transcript']:
        st.markdown("---")
        st.subheader(f"Raw Transcript: *{st.session_state.get('transcription_file_name', 'Audio')}*")
        st.info(f"Duration: {st.session_state.get('transcription_duration', 0):.2f}s | Language: {st.session_state.get('transcription_language', 'N/A').upper()}")
        st.text_area(
            "Full Lecture Transcript", 
            st.session_state['full_transcript'], 
            height=300, 
            key="display_transcript"
        )
        
        st.download_button(
            label="Download Full Transcript (TXT)",
            data=st.session_state['full_transcript'],
            file_name="full_lecture_transcript.txt",
            mime="text/plain"
        )


# --- 8. Main Application Logic ---
def main():
    
    # --- Session State Initialization ---
    if 'user_db' not in st.session_state:
        # User database stored in session state (insecure but simple simulation)
        st.session_state['user_db'] = {}
    if 'is_authenticated' not in st.session_state:
        st.session_state['is_authenticated'] = False
    if 'current_user' not in st.session_state:
        st.session_state['current_user'] = None
    if 'full_transcript' not in st.session_state:
        st.session_state['full_transcript'] = ""
    if 'summarized_notes' not in st.session_state:
        st.session_state['summarized_notes'] = ""
    if 'quiz_data' not in st.session_state:
        st.session_state['quiz_data'] = None
    if 'flashcard_data' not in st.session_state:
        st.session_state['flashcard_data'] = None
    if 'cards_marked' not in st.session_state:
        st.session_state['cards_marked'] = {}
    
    st.title("🗣️ Lecture Voice-to-Notes & Quiz Generator")
    st.markdown("Convert long audio recordings or files into detailed lecture notes and test your retention.")
    
    with st.sidebar:
        
        st.header("👤 Account Access")
        
        # --- Authentication Logic in Sidebar ---
        if st.session_state['is_authenticated']:
            st.success(f"Logged in as: **{st.session_state['current_user']}**")
            st.button("Logout", on_click=logout, key="logout_btn")
            
        else:
            auth_mode = st.radio(
                "Mode:", 
                ["Login", "Create Account"], 
                key="auth_mode_radio"
            )
            
            with st.form("auth_form"):
                auth_username = st.text_input("Username", key="auth_username")
                auth_password = st.text_input("Password", type="password", key="auth_password")
                
                if auth_mode == "Login":
                    login_submitted = st.form_submit_button("Log In", type="primary")
                    if login_submitted:
                        user_login(auth_username, auth_password)
                else:
                    create_submitted = st.form_submit_button("Create Account", type="primary")
                    if create_submitted:
                        create_account(auth_username, auth_password)

        st.markdown("---")
        st.header("🛠️ Configuration")
        
        # --- Transcription Model Selection ---
        model_choice = st.selectbox(
            "Select Transcription Model:",
            ["base", "small", "medium"],
            index=0,
            help="Smaller models are faster but less accurate. 'Base' is a good balance for CPU use."
        )
        
        if API_KEY == "AIzaSyBPW-iSeuA2Ze0hv8268PmUmmd4gaJiJaU":
            st.warning("⚠️ **SECURITY ALERT:** API key is hardcoded. Please use `st.secrets` for production.")
            
        st.markdown("---")
        st.caption("App built with Streamlit, Faster-Whisper, and Google Gemini.")

    # --- Main Tab Navigation ---
    tab_names = ['Upload Audio File', 'Summarized Notes', 'Review Quiz', 'Flashcard Study']
    
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
    
    with tab1:
        audio_upload_tab(model_choice)
        
    with tab2:
        notes_generation_tab()
        
    with tab3:
        quiz_generation_tab()
        
    with tab4:
        flashcard_study_tab()


if __name__ == "__main__":
    main()