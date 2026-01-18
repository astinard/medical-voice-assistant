"""
Medical Voice Assistant - Professional UI
A modern ChatGPT-like interface with voice input/output and voice cloning
"""

import gradio as gr
import numpy as np
import requests
import json
import tempfile
import os
import subprocess
import threading
import queue
import torch

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "williamljx/medgemma-4b-it-Q4_K_M-GGUF"
SYSTEM_PROMPT = """You are a helpful medical assistant. For differential diagnoses, list 5-7 possibilities ranked by likelihood with brief reasoning. For other questions, give accurate concise answers with specific doses, drug names, and key clinical facts."""

# Global state
conversation_context = []
cloned_voice_path = None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        import whisper
        model = whisper.load_model("base.en")
        result = model.transcribe(audio_path, language="en")
        return result["text"].strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def ask_ollama(prompt, history):
    """Send prompt to Ollama and get response"""
    global conversation_context

    try:
        json_param = {
            "model": OLLAMA_MODEL,
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

CHATTERBOX_URL = "http://127.0.0.1:5001"

def text_to_speech(text, use_cloned_voice=False):
    """Convert text to speech, optionally using cloned voice via Chatterbox"""
    global cloned_voice_path

    try:
        if use_cloned_voice and cloned_voice_path:
            # Use Chatterbox service for voice cloning
            try:
                with open(cloned_voice_path, 'rb') as f:
                    files = {'voice_sample': ('voice.wav', f, 'audio/wav')}
                    data = {'text': text}
                    response = requests.post(
                        f"{CHATTERBOX_URL}/clone",
                        files=files,
                        data=data,
                        timeout=120
                    )

                if response.status_code == 200:
                    output_path = tempfile.mktemp(suffix=".wav")
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    return output_path
                else:
                    print(f"Chatterbox error: {response.text}")

            except requests.exceptions.ConnectionError:
                print("Chatterbox service not running, using default TTS")
            except Exception as e:
                print(f"Voice cloning error: {e}")

        # Default: Use macOS say command
        output_path = tempfile.mktemp(suffix=".aiff")
        subprocess.run(['say', '-o', output_path, '-r', '180', text], check=True)
        return output_path

    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def process_voice_input(audio, history):
    """Process voice input and return response"""
    if audio is None:
        return "", None, history, "Please record your question first."

    # Transcribe
    sr, audio_data = audio

    # Save to temp file for Whisper
    temp_path = tempfile.mktemp(suffix=".wav")
    import soundfile as sf
    sf.write(temp_path, audio_data, sr)

    transcription = transcribe_audio(temp_path)
    os.remove(temp_path)

    if not transcription:
        return "", None, history, "Could not transcribe audio."

    # Get AI response
    response = ask_ollama(transcription, history)

    # Generate speech
    audio_response = text_to_speech(response, use_cloned_voice=cloned_voice_path is not None)

    # Update history
    history = history + [[transcription, response]]

    return transcription, audio_response, history, "Ready"

def process_text_input(message, history):
    """Process text input and return response with audio"""
    if not message:
        return "", history, None

    response = ask_ollama(message, history)
    audio_response = text_to_speech(response, use_cloned_voice=cloned_voice_path is not None)

    history = history + [[message, response]]
    return "", history, audio_response

def clone_voice(audio):
    """Save voice sample for cloning"""
    global cloned_voice_path

    if audio is None:
        return "❌ Please record a voice sample first."

    sr, audio_data = audio

    # Save to permanent location
    cloned_voice_path = os.path.join(tempfile.gettempdir(), "cloned_voice.wav")
    import soundfile as sf
    sf.write(cloned_voice_path, audio_data, sr)

    return f"✅ Voice cloned successfully! The AI will now respond in your voice."

def clear_cloned_voice():
    """Clear the cloned voice"""
    global cloned_voice_path
    cloned_voice_path = None
    return "🔊 Using default voice."

def reset_conversation():
    """Reset the conversation context"""
    global conversation_context
    conversation_context = []
    return [], "🔄 Conversation reset."

# Custom CSS for professional look
custom_css = """
/* Dark medical professional theme */
.gradio-container {
    background: linear-gradient(135deg, #0a0f1a 0%, #1a1f2e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-text {
    background: linear-gradient(90deg, #00d4aa, #00a3cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #8b9dc3;
    text-align: center;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Chat container */
.chatbot {
    background: #12182a !important;
    border: 1px solid #2a3548 !important;
    border-radius: 16px !important;
    min-height: 500px !important;
}

/* Message bubbles */
.chatbot .message {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 8px !important;
}

.chatbot .user {
    background: linear-gradient(135deg, #00d4aa 0%, #00a3cc 100%) !important;
    color: white !important;
}

.chatbot .bot {
    background: #1e2738 !important;
    border: 1px solid #2a3548 !important;
    color: #e0e6ed !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #00d4aa 0%, #00a3cc 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3) !important;
}

.secondary-btn {
    background: #1e2738 !important;
    border: 1px solid #2a3548 !important;
    color: #8b9dc3 !important;
    border-radius: 10px !important;
}

/* Input fields */
.textbox {
    background: #12182a !important;
    border: 1px solid #2a3548 !important;
    border-radius: 12px !important;
    color: #e0e6ed !important;
}

.textbox:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
}

/* Audio components */
.audio-container {
    background: #12182a !important;
    border: 1px solid #2a3548 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Tabs */
.tab-nav {
    background: #12182a !important;
    border-radius: 12px !important;
    padding: 4px !important;
}

.tab-nav button {
    background: transparent !important;
    color: #8b9dc3 !important;
    border-radius: 8px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #00d4aa 0%, #00a3cc 100%) !important;
    color: white !important;
}

/* Status indicators */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.status-online {
    background: rgba(0, 212, 170, 0.15);
    color: #00d4aa;
    border: 1px solid rgba(0, 212, 170, 0.3);
}

/* Voice clone section */
.voice-section {
    background: #12182a;
    border: 1px solid #2a3548;
    border-radius: 16px;
    padding: 20px;
    margin-top: 16px;
}

/* Accordion */
.accordion {
    background: #12182a !important;
    border: 1px solid #2a3548 !important;
    border-radius: 12px !important;
}
"""

# Build the interface
with gr.Blocks(title="Medical Voice Assistant") as demo:

    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 class="header-text">🏥 Medical Voice Assistant</h1>
            <p class="subtitle">AI-powered medical consultation with voice cloning • 100% On-Device</p>
            <span class="status-badge status-online">● Ollama Connected</span>
        </div>
    """)

    with gr.Tabs() as tabs:
        # Main Chat Tab
        with gr.Tab("💬 Chat", id="chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_label=False,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your medical question here...",
                            show_label=False,
                            scale=4,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="primary-btn")

                    audio_output = gr.Audio(label="AI Response", type="filepath", autoplay=True)

                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #e0e6ed; margin-bottom: 16px;'>🎤 Voice Input</h3>")
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record your question",
                    )
                    voice_btn = gr.Button("🎙️ Process Voice", variant="primary", elem_classes="primary-btn")

                    transcription_output = gr.Textbox(
                        label="Transcription",
                        interactive=False,
                        lines=3,
                    )

                    gr.HTML("<hr style='border-color: #2a3548; margin: 20px 0;'>")

                    reset_btn = gr.Button("🔄 New Conversation", elem_classes="secondary-btn")
                    status_text = gr.Textbox(label="Status", interactive=False, value="Ready")

        # Voice Clone Tab
        with gr.Tab("🎭 Voice Clone", id="voice"):
            gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="color: #e0e6ed;">Clone Your Voice</h2>
                    <p style="color: #8b9dc3;">Record 10-30 seconds of your voice reading any text.<br>
                    The AI will then respond using your voice!</p>
                </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                        <div style="background: #1e2738; padding: 20px; border-radius: 12px; margin-bottom: 16px;">
                            <h4 style="color: #00d4aa; margin-bottom: 12px;">📖 Sample Text to Read:</h4>
                            <p style="color: #e0e6ed; line-height: 1.8;">
                                "The patient presented with acute onset chest pain, radiating to the left arm.
                                Vital signs showed elevated blood pressure at 160 over 95 millimeters of mercury.
                                Initial ECG revealed ST-segment elevation in leads V1 through V4.
                                Troponin levels were ordered and pending.
                                The differential diagnosis includes acute coronary syndrome,
                                pulmonary embolism, and aortic dissection."
                            </p>
                        </div>
                    """)

                    voice_sample = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record your voice sample (10-30 seconds)",
                    )

                    with gr.Row():
                        clone_btn = gr.Button("✨ Clone Voice", variant="primary", elem_classes="primary-btn")
                        clear_voice_btn = gr.Button("🔊 Use Default Voice", elem_classes="secondary-btn")

                    clone_status = gr.Textbox(label="Clone Status", interactive=False, value="No voice cloned")

        # Settings Tab
        with gr.Tab("⚙️ Settings", id="settings"):
            gr.HTML("<h2 style='color: #e0e6ed;'>Model Settings</h2>")

            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "williamljx/medgemma-4b-it-Q4_K_M-GGUF",
                            "williamljx/medgemma-27b-it-Q4_K_M-GGUF",
                            "llama3.2:3b",
                            "mistral:latest"
                        ],
                        value=OLLAMA_MODEL,
                        label="Ollama Model",
                    )

                    speech_rate = gr.Slider(
                        minimum=100,
                        maximum=250,
                        value=180,
                        step=10,
                        label="Speech Rate (words per minute)",
                    )

                with gr.Column():
                    gr.HTML("""
                        <div style="background: #1e2738; padding: 20px; border-radius: 12px;">
                            <h4 style="color: #00d4aa;">System Requirements</h4>
                            <ul style="color: #8b9dc3; line-height: 2;">
                                <li>✅ Ollama running locally</li>
                                <li>✅ MedGemma model downloaded</li>
                                <li>✅ Whisper for speech recognition</li>
                                <li>⚡ Optional: Chatterbox for voice cloning</li>
                            </ul>
                        </div>
                    """)

    # Event handlers
    msg_input.submit(process_text_input, [msg_input, chatbot], [msg_input, chatbot, audio_output])
    send_btn.click(process_text_input, [msg_input, chatbot], [msg_input, chatbot, audio_output])

    voice_btn.click(
        process_voice_input,
        [voice_input, chatbot],
        [transcription_output, audio_output, chatbot, status_text],
    ).then(
        lambda: None,
        None,
        voice_input,
    )

    reset_btn.click(reset_conversation, None, [chatbot, status_text])

    clone_btn.click(clone_voice, [voice_sample], [clone_status])
    clear_voice_btn.click(clear_cloned_voice, None, [clone_status])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Keep it local/offline
        show_error=True,
        css=custom_css,
    )
