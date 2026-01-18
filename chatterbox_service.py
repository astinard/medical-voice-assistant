"""
Chatterbox Voice Cloning Service
Run this in the Python 3.11 venv with Chatterbox installed
"""

import os
import sys
import json
import tempfile
import torch
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

# Global model - loaded once
model = None

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model():
    """Load Chatterbox model"""
    global model
    if model is None:
        print("Loading Chatterbox model...")
        from chatterbox.tts import ChatterboxTTS
        device = get_device()
        print(f"Using device: {device}")
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded successfully!")
    return model

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/clone', methods=['POST'])
def clone_voice():
    """
    Clone voice and generate speech

    Expects:
    - text: Text to speak
    - voice_sample: Audio file (WAV) of the voice to clone

    Returns:
    - WAV audio file with cloned voice
    """
    try:
        tts = load_model()

        text = request.form.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Handle voice sample
        voice_sample = request.files.get('voice_sample')
        voice_path = None

        if voice_sample:
            # Save uploaded voice sample
            voice_path = tempfile.mktemp(suffix=".wav")
            voice_sample.save(voice_path)

        # Generate speech
        print(f"Generating speech for: {text[:50]}...")

        if voice_path:
            wav = tts.generate(text, audio_prompt_path=voice_path)
        else:
            wav = tts.generate(text)

        # Save output
        output_path = tempfile.mktemp(suffix=".wav")
        import torchaudio
        torchaudio.save(output_path, wav, tts.sr)

        # Clean up voice sample
        if voice_path and os.path.exists(voice_path):
            os.remove(voice_path)

        print("Speech generated successfully!")
        return send_file(output_path, mimetype='audio/wav')

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_speech():
    """
    Generate speech without cloning (default voice)

    Expects:
    - text: Text to speak

    Returns:
    - WAV audio file
    """
    try:
        tts = load_model()

        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"Generating speech for: {text[:50]}...")
        wav = tts.generate(text)

        output_path = tempfile.mktemp(suffix=".wav")
        import torchaudio
        torchaudio.save(output_path, wav, tts.sr)

        print("Speech generated successfully!")
        return send_file(output_path, mimetype='audio/wav')

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Chatterbox Voice Cloning Service...")
    print(f"Device: {get_device()}")

    # Pre-load model
    load_model()

    # Start server
    app.run(host='127.0.0.1', port=5001, debug=False)
