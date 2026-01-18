"""
Medical Voice Assistant - Unified App
Everything in one: Whisper STT, Ollama LLM, Chatterbox Voice Cloning
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
        # Turbo on CPU: fast (1 step vs 1000) and stable
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

def clean_text_for_speech(text):
    """Clean text for TTS - remove markdown, special chars, etc."""
    import re

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'#{1,6}\s*', '', text)           # # headers
    text = re.sub(r'`([^`]+)`', r'\1', text)        # `code`

    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)

    # Remove special characters that cause issues
    text = re.sub(r'[<>{}[\]|\\^~]', '', text)

    # Clean up whitespace
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Limit length for TTS (long text = long generation time)
    if len(text) > 500:
        text = text[:500] + "..."

    return text

def text_to_speech(text, use_cloned_voice=False):
    """Convert text to speech, optionally using cloned voice"""
    global cloned_voice_path

    try:
        if use_cloned_voice and cloned_voice_path:
            # Use Chatterbox for voice cloning
            try:
                tts = load_chatterbox()

                # Clean text for speech
                clean_text = clean_text_for_speech(text)
                print(f"Generating cloned speech for: {clean_text[:80]}...")

                wav = tts.generate(clean_text, audio_prompt_path=cloned_voice_path)

                output_path = tempfile.mktemp(suffix=".wav")
                torchaudio.save(output_path, wav, tts.sr)
                print("Cloned speech generated!")
                return output_path

            except Exception as e:
                print(f"Voice cloning error: {e}")
                # Fallback to default

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
        return "❌ Please record a voice sample first.", gr.update()

    sr, audio_data = audio

    # Pre-load Chatterbox while saving
    print("Pre-loading Chatterbox for voice cloning...")

    cloned_voice_path = os.path.join(tempfile.gettempdir(), "cloned_voice.wav")
    import soundfile as sf
    sf.write(cloned_voice_path, audio_data, sr)

    # Load model in background
    try:
        load_chatterbox()
        return "✅ Voice cloned! AI will respond in your voice.", gr.update(value="🎭 Using Cloned Voice")
    except Exception as e:
        return f"❌ Error loading voice cloning: {e}", gr.update()

def clear_cloned_voice():
    """Clear the cloned voice"""
    global cloned_voice_path
    cloned_voice_path = None
    return "🔊 Using default voice.", gr.update(value="🔊 Default Voice")

def reset_conversation():
    """Reset the conversation context"""
    global conversation_context
    conversation_context = []
    return [], "🔄 Conversation reset."

# Build the interface
with gr.Blocks(title="Medical Voice Assistant") as demo:

    gr.HTML("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #1a1f2e 0%, #0a0f1a 100%); border-radius: 16px; margin-bottom: 20px;">
            <h1 style="background: linear-gradient(90deg, #00d4aa, #00a3cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin: 0;">
                🏥 Medical Voice Assistant
            </h1>
            <p style="color: #8b9dc3; margin-top: 10px;">
                AI-powered medical consultation with voice cloning • 100% Offline
            </p>
        </div>
    """)

    with gr.Row():
        voice_status = gr.Textbox(value="🔊 Default Voice", label="Voice Mode", interactive=False, scale=1)

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=450, show_label=False)

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your medical question...",
                            show_label=False,
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    audio_output = gr.Audio(label="AI Response", type="filepath", autoplay=True)

                with gr.Column(scale=1):
                    gr.HTML("<h3>🎤 Voice Input</h3>")
                    voice_input = gr.Audio(sources=["microphone"], type="numpy", label="Record")
                    voice_btn = gr.Button("🎙️ Process Voice", variant="primary")
                    transcription_output = gr.Textbox(label="Transcription", interactive=False, lines=3)

                    gr.HTML("<hr style='margin: 20px 0;'>")
                    reset_btn = gr.Button("🔄 New Conversation")
                    status_text = gr.Textbox(label="Status", interactive=False, value="Ready")

        with gr.Tab("🎭 Voice Clone"):
            gr.Markdown("""
            ## Clone Your Voice
            Record 10-30 seconds of speech. The AI will respond using your voice!

            **✨ Works 100% offline after first load**
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📖 Sample Text to Read:

                    > "The patient presented with acute onset chest pain, radiating to the left arm.
                    > Vital signs showed elevated blood pressure at 160 over 95 millimeters of mercury.
                    > Initial ECG revealed ST-segment elevation in leads V1 through V4.
                    > Troponin levels were ordered and pending. The differential diagnosis includes
                    > acute coronary syndrome, pulmonary embolism, and aortic dissection."
                    """)

                    voice_sample = gr.Audio(sources=["microphone"], type="numpy", label="Record your voice (10-30 sec)")

                    with gr.Row():
                        clone_btn = gr.Button("✨ Clone Voice", variant="primary")
                        clear_voice_btn = gr.Button("🔊 Use Default Voice")

                    clone_status = gr.Textbox(label="Status", interactive=False, value="No voice cloned")

        with gr.Tab("⚙️ Settings"):
            gr.HTML("<h2>Model Settings</h2>")
            model_dropdown = gr.Dropdown(
                choices=[
                    "williamljx/medgemma-4b-it-Q4_K_M-GGUF",
                    "williamljx/medgemma-27b-it-Q4_K_M-GGUF",
                    "llama3.2:3b",
                ],
                value=OLLAMA_MODEL,
                label="Ollama Model",
            )

            gr.HTML(f"""
                <div style="background: #e8f5e9; padding: 20px; border-radius: 12px; margin-top: 20px;">
                    <h4 style="color: #2e7d32;">System Info</h4>
                    <ul>
                        <li><strong>Device:</strong> {get_device().upper()}</li>
                        <li><strong>Whisper:</strong> base.en (loaded on demand)</li>
                        <li><strong>Voice Cloning:</strong> Chatterbox (loaded on demand)</li>
                        <li><strong>Offline:</strong> ✅ Yes, after models downloaded</li>
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
    ).then(lambda: None, None, voice_input)

    reset_btn.click(reset_conversation, None, [chatbot, status_text])

    clone_btn.click(clone_voice, [voice_sample], [clone_status, voice_status])
    clear_voice_btn.click(clear_cloned_voice, None, [clone_status, voice_status])

if __name__ == "__main__":
    print(f"Device: {get_device()}")
    print("Starting Medical Voice Assistant...")
    print("Models will load on first use.")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
