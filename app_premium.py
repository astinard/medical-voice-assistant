"""
Medical Voice Assistant - Premium Edition
Whisper STT, Ollama LLM, Chatterbox Voice Cloning
100% Offline capable
"""

import gradio as gr
import numpy as np
import requests
import json
import tempfile
import os
import subprocess
import torch
import torchaudio

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "williamljx/medgemma-4b-it-Q4_K_M-GGUF"
SYSTEM_PROMPT = """You are a helpful medical assistant. For differential diagnoses, list 5-7 possibilities ranked by likelihood with brief reasoning. For other questions, give accurate concise answers with specific doses, drug names, and key clinical facts."""

# Global state
conversation_context = []
cloned_voice_path = None
whisper_model = None
chatterbox_model = None

# Premium CSS Theme
PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0f1419;
    --bg-secondary: #1a1f2e;
    --bg-tertiary: #252d3d;
    --bg-card: #1e2533;
    --text-primary: #f7f8f8;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-coral: #e8927c;
    --accent-coral-light: #f0a896;
    --accent-teal: #5eead4;
    --accent-blue: #60a5fa;
    --border-subtle: rgba(255,255,255,0.06);
    --border-medium: rgba(255,255,255,0.1);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
    --shadow-glow: 0 0 40px rgba(232,146,124,0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --radius-xl: 28px;
}

* {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at 30% 20%, rgba(232,146,124,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(94,234,212,0.05) 0%, transparent 50%);
    pointer-events: none;
}

.header-container::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.03;
    pointer-events: none;
}

.logo-mark {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, var(--accent-coral) 0%, var(--accent-coral-light) 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-md), var(--shadow-glow);
}

.header-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 42px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.header-subtitle {
    font-size: 16px;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(94,234,212,0.1);
    border: 1px solid rgba(94,234,212,0.2);
    color: var(--accent-teal);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 16px;
}

.header-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    background: var(--accent-teal);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* Control Bar */
.control-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
}

.control-group {
    flex: 1;
}

.control-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* Radio Buttons */
.gr-radio, .gr-checkbox {
    background: transparent !important;
    border: none !important;
}

.gr-radio label, .gr-checkbox label {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 18px !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-radio label:hover, .gr-checkbox label:hover {
    background: var(--bg-card) !important;
    border-color: var(--accent-coral) !important;
    color: var(--text-primary) !important;
}

.gr-radio label.selected, .gr-checkbox label.selected,
.gr-radio input:checked + label, .gr-checkbox input:checked + label {
    background: linear-gradient(135deg, rgba(232,146,124,0.15) 0%, rgba(232,146,124,0.05) 100%) !important;
    border-color: var(--accent-coral) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 0 20px rgba(232,146,124,0.1) !important;
}

/* Tabs */
.gr-tab-nav {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 24px !important;
}

.gr-tab-nav button {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 28px !important;
    color: var(--text-secondary) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.gr-tab-nav button:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

.gr-tab-nav button.selected {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Chat Container */
.chat-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 24px;
    min-height: 500px;
}

/* Chatbot */
.gr-chatbot {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px !important;
}

.gr-chatbot .message {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 20px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

.gr-chatbot .user {
    background: linear-gradient(135deg, rgba(232,146,124,0.12) 0%, rgba(232,146,124,0.06) 100%) !important;
    border-color: rgba(232,146,124,0.2) !important;
}

.gr-chatbot .bot {
    background: var(--bg-tertiary) !important;
}

/* Input Fields */
.gr-textbox textarea, .gr-textbox input {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    padding: 16px 20px !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}

.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: var(--accent-coral) !important;
    box-shadow: 0 0 0 3px rgba(232,146,124,0.1) !important;
    outline: none !important;
}

.gr-textbox textarea::placeholder, .gr-textbox input::placeholder {
    color: var(--text-muted) !important;
}

/* Buttons */
.gr-button {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    padding: 14px 24px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-button:hover {
    background: var(--bg-card) !important;
    border-color: var(--accent-coral) !important;
    transform: translateY(-1px) !important;
}

.gr-button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-coral) 0%, var(--accent-coral-light) 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: var(--shadow-md), 0 0 20px rgba(232,146,124,0.2) !important;
}

.gr-button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg), 0 0 30px rgba(232,146,124,0.3) !important;
}

/* Audio Components */
.gr-audio {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
}

.gr-audio audio {
    width: 100% !important;
}

/* Voice Input Card */
.voice-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
}

.voice-card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.mic-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent-coral) 0%, var(--accent-coral-light) 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

/* Status Display */
.status-display {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    font-family: 'DM Sans', monospace !important;
    font-size: 13px;
    color: var(--text-secondary);
}

.status-display.success {
    border-color: rgba(94,234,212,0.3);
    color: var(--accent-teal);
}

/* Clone Tab */
.clone-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 40px;
}

.sample-text-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-medium);
    border-left: 3px solid var(--accent-coral);
    border-radius: var(--radius-md);
    padding: 24px;
    margin: 24px 0;
}

.sample-text-card p {
    font-size: 16px;
    line-height: 1.8;
    color: var(--text-secondary);
    font-style: italic;
}

/* Settings Tab */
.settings-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 40px;
}

.info-card {
    background: linear-gradient(135deg, rgba(94,234,212,0.08) 0%, rgba(94,234,212,0.02) 100%);
    border: 1px solid rgba(94,234,212,0.15);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-top: 24px;
}

.info-card h4 {
    color: var(--accent-teal);
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

.info-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.info-card li {
    padding: 8px 0;
    color: var(--text-secondary);
    font-size: 14px;
    border-bottom: 1px solid var(--border-subtle);
}

.info-card li:last-child {
    border-bottom: none;
}

.info-card strong {
    color: var(--text-primary);
}

/* Labels */
label, .gr-label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Dropdown */
.gr-dropdown {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--bg-tertiary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Responsive */
@media (max-width: 768px) {
    .header-title {
        font-size: 28px;
    }

    .control-bar {
        flex-direction: column;
    }
}
"""

def get_device():
    """Get best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_whisper():
    """Load Whisper model (lazy loading)"""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        import whisper
        whisper_model = whisper.load_model("base.en")
        print("Whisper loaded!")
    return whisper_model

def load_chatterbox():
    """Load Chatterbox Turbo model on CPU - fast and stable"""
    global chatterbox_model
    if chatterbox_model is None:
        print("Loading Chatterbox Turbo on CPU...")
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        chatterbox_model = ChatterboxTurboTTS.from_pretrained(device="cpu")
        print("Chatterbox Turbo loaded!")
    return chatterbox_model

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper()
        result = model.transcribe(audio_path, language="en")
        return result["text"].strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def ask_ollama(prompt, history, model=None):
    """Send prompt to Ollama and get response"""
    global conversation_context, OLLAMA_MODEL
    use_model = model if model else OLLAMA_MODEL

    try:
        json_param = {
            "model": use_model,
            "stream": False,
            "context": conversation_context,
            "prompt": prompt,
            "system": SYSTEM_PROMPT
        }

        response = requests.post(
            OLLAMA_URL,
            json=json_param,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        conversation_context = result.get('context', [])
        return result.get('response', 'No response received')

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"

def clean_text_for_speech(text):
    """Clean text for TTS"""
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[<>{}[\]|\\^~]', '', text)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if len(text) > 400:
        text = text[:400] + "..."
    return text

def text_to_speech(text, use_cloned_voice=False):
    """Convert text to speech"""
    global cloned_voice_path

    try:
        if use_cloned_voice and cloned_voice_path:
            try:
                tts = load_chatterbox()
                clean_text = clean_text_for_speech(text)
                print(f"Generating cloned speech for: {clean_text[:80]}...")
                wav = tts.generate(clean_text, audio_prompt_path=cloned_voice_path)
                output_path = tempfile.mktemp(suffix=".wav")
                torchaudio.save(output_path, wav, tts.sr)
                print("Cloned speech generated!")
                return output_path
            except Exception as e:
                print(f"Voice cloning error: {e}")

        clean_text = clean_text_for_speech(text)
        print(f"Generating speech for: {clean_text[:80]}...")
        output_path = tempfile.mktemp(suffix=".aiff")
        subprocess.run(['say', '-o', output_path, '-r', '180', clean_text], check=True)
        print("Speech generated!")
        return output_path

    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def process_voice_input(audio, history, voice_mode, model_choice):
    """Process voice input and return response"""
    import time

    if audio is None:
        return "", None, history, "⏳ Please record your question first"

    total_start = time.time()
    sr, audio_data = audio

    temp_path = tempfile.mktemp(suffix=".wav")
    import soundfile as sf
    sf.write(temp_path, audio_data, sr)

    stt_start = time.time()
    transcription = transcribe_audio(temp_path)
    stt_time = time.time() - stt_start
    os.remove(temp_path)

    if not transcription:
        return "", None, history, "⚠️ Could not transcribe audio"

    llm_start = time.time()
    response = ask_ollama(transcription, history, model=model_choice)
    llm_time = time.time() - llm_start

    tts_start = time.time()
    use_cloned = (voice_mode == "My Voice") and (cloned_voice_path is not None)
    audio_response = text_to_speech(response, use_cloned_voice=use_cloned)
    tts_time = time.time() - tts_start

    total_time = time.time() - total_start
    status = f"✓ {total_time:.1f}s total  ·  STT {stt_time:.1f}s  ·  LLM {llm_time:.1f}s  ·  TTS {tts_time:.1f}s"

    history = history + [[transcription, response]]
    return transcription, audio_response, history, status

def process_text_input(message, history, voice_mode, model_choice):
    """Process text input and return response with audio"""
    import time

    if not message:
        return "", history, None, "Ready"

    total_start = time.time()

    llm_start = time.time()
    response = ask_ollama(message, history, model=model_choice)
    llm_time = time.time() - llm_start

    tts_start = time.time()
    use_cloned = (voice_mode == "My Voice") and (cloned_voice_path is not None)
    audio_response = text_to_speech(response, use_cloned_voice=use_cloned)
    tts_time = time.time() - tts_start

    total_time = time.time() - total_start
    status = f"✓ {total_time:.1f}s total  ·  LLM {llm_time:.1f}s  ·  TTS {tts_time:.1f}s"

    history = history + [[message, response]]
    return "", history, audio_response, status

def clone_voice(audio):
    """Save voice sample for cloning"""
    global cloned_voice_path

    if audio is None:
        return "⏳ Please record a voice sample first", gr.update()

    sr, audio_data = audio
    print("Pre-loading Chatterbox for voice cloning...")

    cloned_voice_path = os.path.join(tempfile.gettempdir(), "cloned_voice.wav")
    import soundfile as sf
    sf.write(cloned_voice_path, audio_data, sr)

    try:
        load_chatterbox()
        return "✓ Voice cloned successfully! Select 'My Voice' to use it.", gr.update(value="Voice Ready")
    except Exception as e:
        return f"⚠️ Error: {e}", gr.update()

def clear_cloned_voice():
    """Clear the cloned voice"""
    global cloned_voice_path
    cloned_voice_path = None
    return "Voice cleared. Using system voice.", gr.update(value="No Voice")

def reset_conversation():
    """Reset the conversation context"""
    global conversation_context
    conversation_context = []
    return [], "Conversation cleared"

# Quick Chat - separate context for fast text-only chat
quick_chat_context = []

def process_quick_chat(message, history, model_choice):
    """Process text-only chat - fastest mode, no audio"""
    global quick_chat_context
    import time

    if not message:
        return "", history, "Ready"

    start = time.time()

    try:
        json_param = {
            "model": model_choice,
            "stream": False,
            "context": quick_chat_context,
            "prompt": message,
            "system": SYSTEM_PROMPT
        }

        response = requests.post(
            OLLAMA_URL,
            json=json_param,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        quick_chat_context = result.get('context', [])
        response_text = result.get('response', 'No response received')

    except requests.exceptions.RequestException as e:
        response_text = f"Error connecting to Ollama: {str(e)}"

    elapsed = time.time() - start
    status = f"✓ {elapsed:.2f}s"
    history = history + [[message, response_text]]
    return "", history, status

def reset_quick_chat():
    """Reset quick chat context"""
    global quick_chat_context
    quick_chat_context = []
    return [], "Chat cleared"

# Build the interface
with gr.Blocks(title="Medica - AI Medical Assistant") as demo:

    # Premium Header
    gr.HTML("""
        <div class="header-container">
            <div class="logo-mark">🩺</div>
            <h1 class="header-title">Medica</h1>
            <p class="header-subtitle">AI-powered medical consultation with voice cloning technology</p>
            <div class="header-badge">
                <span>100% Offline</span>
            </div>
        </div>
    """)

    # Control Bar
    with gr.Row(elem_classes="control-bar"):
        voice_status = gr.Textbox(
            value="No Voice",
            label="VOICE STATUS",
            interactive=False,
            scale=1,
            container=True
        )
        voice_mode = gr.Radio(
            choices=["Fast", "My Voice"],
            value="Fast",
            label="RESPONSE MODE",
            scale=2
        )
        model_choice = gr.Radio(
            choices=["williamljx/medgemma-4b-it-Q4_K_M-GGUF", "alibayram/medgemma:27b"],
            value="williamljx/medgemma-4b-it-Q4_K_M-GGUF",
            label="MODEL",
            scale=2
        )

    # Main Content
    with gr.Tabs():
        with gr.Tab("Consultation", id="chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=480,
                        show_label=False,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Describe symptoms or ask a medical question...",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    audio_output = gr.Audio(
                        label="AI Response",
                        type="filepath",
                        autoplay=True,
                        show_label=False,
                    )

                with gr.Column(scale=1):
                    gr.HTML("""
                        <div class="voice-card">
                            <div class="voice-card-title">
                                <div class="mic-icon">🎤</div>
                                Voice Input
                            </div>
                        </div>
                    """)

                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="",
                        show_label=False,
                    )
                    voice_btn = gr.Button("Process Voice", variant="primary")
                    transcription_output = gr.Textbox(
                        label="Transcription",
                        interactive=False,
                        lines=2,
                        show_label=True,
                    )

                    gr.HTML("<div style='height: 16px'></div>")

                    reset_btn = gr.Button("Clear Conversation", variant="secondary")
                    status_text = gr.Textbox(
                        label="Performance",
                        interactive=False,
                        value="Ready",
                        show_label=True,
                    )

        with gr.Tab("Quick Chat", id="quick"):
            gr.HTML("""
                <div style="padding: 16px 0;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #5eead4 0%, #2dd4bf 100%);
                                    border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">
                            ⚡
                        </div>
                        <h2 style="font-family: 'Playfair Display', serif; font-size: 24px; color: #f7f8f8; margin: 0;">
                            Quick Chat
                        </h2>
                    </div>
                    <p style="color: #6b7280; font-size: 14px; margin: 0;">
                        Text-only mode · Fastest response time · No audio processing
                    </p>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=4):
                    quick_chatbot = gr.Chatbot(
                        label="Quick Chat",
                        height=500,
                        show_label=False,
                    )

                    with gr.Row():
                        quick_msg_input = gr.Textbox(
                            placeholder="Type your medical question...",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        quick_send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="background: rgba(94,234,212,0.08); border: 1px solid rgba(94,234,212,0.15);
                                    border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                            <h4 style="color: #5eead4; font-size: 13px; text-transform: uppercase;
                                       letter-spacing: 0.5px; margin: 0 0 12px 0;">Why Quick Chat?</h4>
                            <ul style="color: #9ca3af; font-size: 13px; margin: 0; padding-left: 16px; line-height: 1.8;">
                                <li>No speech-to-text processing</li>
                                <li>No text-to-speech generation</li>
                                <li>Direct LLM interaction</li>
                                <li>~2-5s response time</li>
                            </ul>
                        </div>
                    """)

                    quick_clear_btn = gr.Button("Clear Chat", variant="secondary")
                    quick_status = gr.Textbox(
                        label="Response Time",
                        interactive=False,
                        value="Ready",
                        show_label=True,
                    )

        with gr.Tab("Voice Clone", id="clone"):
            gr.HTML("""
                <div class="clone-container">
                    <h2 style="font-family: 'Playfair Display', serif; font-size: 28px; color: #f7f8f8; margin-bottom: 8px;">
                        Voice Cloning
                    </h2>
                    <p style="color: #9ca3af; font-size: 16px; margin-bottom: 32px;">
                        Record 10-30 seconds of speech. The AI will respond using your voice.
                    </p>

                    <div class="sample-text-card">
                        <p style="margin: 0; color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; font-style: normal;">
                            Sample Text to Read
                        </p>
                        <p>
                            "The patient presented with acute onset chest pain, radiating to the left arm.
                            Vital signs showed elevated blood pressure at 160 over 95 millimeters of mercury.
                            Initial ECG revealed ST-segment elevation in leads V1 through V4.
                            Troponin levels were ordered and pending."
                        </p>
                    </div>
                </div>
            """)

            with gr.Row():
                with gr.Column():
                    voice_sample = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record Your Voice",
                    )

                    with gr.Row():
                        clone_btn = gr.Button("Clone Voice", variant="primary")
                        clear_voice_btn = gr.Button("Clear Voice", variant="secondary")

                    clone_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="No voice cloned"
                    )

        with gr.Tab("Settings", id="settings"):
            gr.HTML(f"""
                <div class="settings-container">
                    <h2 style="font-family: 'Playfair Display', serif; font-size: 28px; color: #f7f8f8; margin-bottom: 24px;">
                        System Information
                    </h2>

                    <div class="info-card">
                        <h4>Runtime Configuration</h4>
                        <ul>
                            <li><strong>Compute Device</strong> — {get_device().upper()}</li>
                            <li><strong>Speech Recognition</strong> — Whisper base.en</li>
                            <li><strong>Voice Synthesis</strong> — Chatterbox Turbo</li>
                            <li><strong>Medical Model</strong> — MedGemma (4B / 27B)</li>
                            <li><strong>Network Status</strong> — 100% Offline Capable</li>
                        </ul>
                    </div>

                    <div style="margin-top: 32px; padding: 20px; background: rgba(232,146,124,0.08); border-radius: 12px; border: 1px solid rgba(232,146,124,0.15);">
                        <p style="color: #e8927c; font-size: 14px; margin: 0;">
                            <strong>Note:</strong> Models are loaded on-demand to optimize memory usage.
                            First request may take longer while models initialize.
                        </p>
                    </div>
                </div>
            """)

    # Event handlers
    msg_input.submit(
        process_text_input,
        [msg_input, chatbot, voice_mode, model_choice],
        [msg_input, chatbot, audio_output, status_text]
    )
    send_btn.click(
        process_text_input,
        [msg_input, chatbot, voice_mode, model_choice],
        [msg_input, chatbot, audio_output, status_text]
    )

    voice_btn.click(
        process_voice_input,
        [voice_input, chatbot, voice_mode, model_choice],
        [transcription_output, audio_output, chatbot, status_text],
    ).then(lambda: None, None, voice_input)

    reset_btn.click(reset_conversation, None, [chatbot, status_text])
    clone_btn.click(clone_voice, [voice_sample], [clone_status, voice_status])
    clear_voice_btn.click(clear_cloned_voice, None, [clone_status, voice_status])

    # Quick Chat event handlers
    quick_msg_input.submit(
        process_quick_chat,
        [quick_msg_input, quick_chatbot, model_choice],
        [quick_msg_input, quick_chatbot, quick_status]
    )
    quick_send_btn.click(
        process_quick_chat,
        [quick_msg_input, quick_chatbot, model_choice],
        [quick_msg_input, quick_chatbot, quick_status]
    )
    quick_clear_btn.click(reset_quick_chat, None, [quick_chatbot, quick_status])

if __name__ == "__main__":
    print(f"Device: {get_device()}")
    print("Starting Medica - Premium Medical Voice Assistant...")
    print("Models will load on first use.")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=PREMIUM_CSS,
    )
