import sys
import json
import wave
import time
import pyttsx3
import torch
import requests
import soundfile
import yaml
import pygame
import pygame.locals
import numpy as np
import pyaudio
import logging
import threading
import queue
import re

# Medical Whisper from HuggingFace
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Flag to use medical whisper vs standard whisper
USE_MEDICAL_WHISPER = False

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
USER_COLOR = (100,200,255)
AI_COLOR = (100,255,150)
REC_SIZE = 80
FONT_SIZE = 18
WIDTH = 800
HEIGHT = 650
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 60
DIALOGUE_MARGIN = 20

INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}
INPUT_CONFIG_PATH ="assistant.yaml"

class Assistant:
    def __init__(self):
        logging.info("Initializing Assistant")
        self.config = self.init_config()

        programIcon = pygame.image.load('assistant.png')

        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Assistant")

        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.audio = pyaudio.PyAudio()

        self.tts = pyttsx3.init("nsss");
        self.tts.setProperty('rate', self.tts.getProperty('rate') - 20)

        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()

        self.display_message(self.config.messages.loadingModel)
        if USE_MEDICAL_WHISPER:
            logging.info("Loading Google MedASR model...")
            from transformers import pipeline
            self.asr_pipeline = pipeline('automatic-speech-recognition', model='google/medasr')
            logging.info("MedASR loaded successfully")
        else:
            import whisper
            self.model = whisper.load_model(self.config.whisperRecognition.modelPath)
        self.context = []

        self.text_to_speech(self.config.conversation.greeting)
        time.sleep(0.5)
        self.display_message(self.config.messages.pressSpace)

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.audio.terminate()
        pygame.quit()
        sys.exit()

    def init_config(self):
        logging.info("Initializing configuration")
        class Inst:
            pass

        with open('assistant.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)

        config = Inst()
        config.messages = Inst()
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

        config.conversation = Inst()
        config.conversation.greeting = configYaml["conversation"]["greeting"]

        config.ollama = Inst()
        config.ollama.url = configYaml["ollama"]["url"]
        config.ollama.model = configYaml["ollama"]["model"]

        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]

        return config

    def display_rec_start(self):
        logging.info("Displaying recording start")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        logging.info(f"Displaying sound energy: {energy}")
        COL_COUNT = 5
        RED_CENTER = 100
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE*energy)
        hspace, vspace = 2*KWIDTH, int(KHEIGHT/2)
        def rect_coords(x, y):
            return (int(x-KWIDTH/2), int(y-KHEIGHT/2),
                    KWIDTH, KHEIGHT)
        for i in range(-int(np.floor(COL_COUNT/2)), int(np.ceil(COL_COUNT/2))):
            x, y, count = WIDTH/2+(i*hspace), HEIGHT/2, amplitude-2*abs(i)

            mid = int(np.ceil(count/2))
            for i in range(0, mid):
                offset = i*(KHEIGHT+vspace)
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y+offset))
                #mirror:
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y-offset))
        pygame.display.flip()

    def display_message(self, text):
        logging.info(f"Displaying message: {text}")
        self.windowSurface.fill(BACK_COLOR)

        label = self.font.render(text
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT - 40))

        # Draw dialogue history
        self.draw_dialogue()
        pygame.display.flip()

    def word_wrap(self, text, max_chars=55):
        """Wrap text to fit within max_chars per line"""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def draw_dialogue(self):
        """Draw conversation history on screen - simplified"""
        if not hasattr(self, 'dialogue_history'):
            self.dialogue_history = []

        y_pos = DIALOGUE_MARGIN
        line_height = FONT_SIZE + 2
        max_lines = (HEIGHT - 70) // line_height

        # Just show last user message and last AI message
        recent = self.dialogue_history[-2:] if len(self.dialogue_history) >= 2 else self.dialogue_history

        all_lines = []
        for speaker, text in recent:
            color = USER_COLOR if speaker == "You" else AI_COLOR
            # First line with speaker label
            wrapped = self.word_wrap(text, 70)
            for i, line in enumerate(wrapped):
                if i == 0:
                    all_lines.append((f"{speaker}: {line}", color))
                else:
                    all_lines.append((f"  {line}", color))
            all_lines.append(("", TEXT_COLOR))  # Blank line between

        # Show what fits
        for display_text, color in all_lines[:max_lines]:
            if display_text:
                label = self.font.render(display_text, 1, color)
                self.windowSurface.blit(label, (DIALOGUE_MARGIN, y_pos))
            y_pos += line_height

    def add_dialogue(self, speaker, text):
        """Add a message to dialogue history"""
        if not hasattr(self, 'dialogue_history'):
            self.dialogue_history = []
        # Strip markdown formatting
        clean_text = text.strip()
        clean_text = clean_text.replace('**', '').replace('*', '')
        clean_text = clean_text.replace('`', '')
        # Remove numbered list formatting
        import re
        clean_text = re.sub(r'^\d+\.\s+', '', clean_text, flags=re.MULTILINE)
        self.dialogue_history.append((speaker, clean_text))

    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:
        logging.info("Capturing waveform from microphone")
        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:
            pygame.event.pump() # process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        logging.info("Converting speech to text")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription")
                if USE_MEDICAL_WHISPER:
                    # MedASR expects 16kHz audio
                    result = self.asr_pipeline(waveform, chunk_length_s=20, stride_length_s=2)
                    text = result['text']
                    # Clean up CTC output - remove epsilon tokens
                    text = text.replace('<epsilon>', '').replace('</s>', '').replace('<s>', '')
                    # Remove excessive dashes/hyphens
                    text = re.sub(r'-{2,}', '-', text)
                    # Remove duplicate consecutive characters within words (e.g., "Alrightight" -> "Alright")
                    def clean_word(word):
                        # Remove repeated syllables at end (e.g., "yearar" -> "year", "oldold" -> "old")
                        for length in range(2, min(5, len(word)//2 + 1)):
                            if len(word) >= length * 2:
                                end = word[-length:]
                                if word[-(length*2):-length] == end:
                                    word = word[:-length]
                        return word
                    words = text.split()
                    cleaned_words = []
                    prev_word = None
                    for word in words:
                        word = clean_word(word)
                        if word != prev_word:
                            cleaned_words.append(word)
                            prev_word = word
                    text = ' '.join(cleaned_words)
                else:
                    transcript = self.model.transcribe(waveform,
                                                    language=self.config.whisperRecognition.lang,
                                                    fp16=torch.cuda.is_available())
                    text = transcript["text"]
                logging.info("Transcription completed")
                print('\nMe:\n', text.strip())
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()


    def ask_ollama(self, prompt, responseCallback):
        logging.info(f"Asking OLLaMa with prompt: {prompt}")
        full_prompt = prompt if hasattr(self, "contextSent") else (prompt)
        self.contextSent = True
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt,
            "system": "You are a helpful medical assistant. For differential diagnoses, list 5-7 possibilities ranked by likelihood with brief reasoning. For other questions, give accurate concise answers with specific doses, drug names, and key clinical facts."
        }
        
        try:
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    stream=True,
                                    timeout=60)  # Increase the timeout value
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                body = json.loads(line)
                token = body.get('response', '')
                full_response += token

                if 'error' in body:
                    logging.error(f"Error from OLLaMa: {body['error']}")
                    responseCallback("Error: " + body['error'])
                    return

                if body.get('done', False) and 'context' in body:
                    self.context = body['context']
                    break

            responseCallback(full_response.strip())

        except requests.exceptions.ReadTimeout as e:
            logging.error(f"ReadTimeout occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, the request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, an error occurred. Please try again.")


    def text_to_speech(self, text):
        logging.info(f"Converting text to speech: {text}")
        print('\nAI:\n', text.strip())

        # Add AI response to dialogue history
        if text.strip():
            self.add_dialogue("AI", text)
            self.display_message("Speaking... (press Space to interrupt)")

        self.is_speaking = True
        self.say_process = None

        def speak():
            try:
                logging.info("Starting speech playback using macOS say")
                import subprocess
                # Use macOS say command - can be killed for interrupt
                self.say_process = subprocess.Popen(['say', '-r', '180', text])
                self.say_process.wait()
                logging.info("Speech playback completed")
            except Exception as e:
                logging.error(f"An error occurred during speech playback: {str(e)}")
            finally:
                self.is_speaking = False
                self.say_process = None

        # Run in thread so main loop can check for interrupts
        self.speech_thread = threading.Thread(target=speak)
        self.speech_thread.start()

    def interrupt_speech(self):
        """Stop current speech"""
        logging.info("Interrupting speech")
        self.is_speaking = False
        if hasattr(self, 'say_process') and self.say_process:
            try:
                self.say_process.terminate()
            except:
                pass

    def wait_for_speech(self):
        """Wait for speech to complete, checking for interrupt"""
        if hasattr(self, 'speech_thread'):
            self.speech_thread.join()


def main():
    logging.info("Starting Assistant")
    pygame.init()

    ass = Assistant()
    ass.is_speaking = False

    push_to_talk_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == push_to_talk_key:
                    # If speaking, interrupt first
                    if ass.is_speaking:
                        logging.info("Interrupting speech for new input")
                        ass.interrupt_speech()
                        time.sleep(0.3)  # Brief pause for TTS to stop

                    logging.info("Push-to-talk key pressed")
                    speech = ass.waveform_from_mic(push_to_talk_key)

                    transcription = ass.speech_to_text(waveform=speech)

                    # Add user message to dialogue and update display
                    if transcription.strip():
                        ass.add_dialogue("You", transcription)
                        ass.display_message("Thinking...")

                    ass.ask_ollama(transcription, ass.text_to_speech)

                    # Wait for speech in background, checking for new input
                    while ass.is_speaking:
                        ass.clock.tick(60)
                        pygame.event.pump()
                        pressed = pygame.key.get_pressed()
                        if pressed[push_to_talk_key]:
                            break  # Will handle interrupt on next loop

                    if not ass.is_speaking:
                        ass.display_message(ass.config.messages.pressSpace)

                elif event.key == quit_key:
                    logging.info("Quit key pressed")
                    ass.shutdown()

            elif event.type == pygame.locals.QUIT:
                ass.shutdown()


if __name__ == "__main__":
    main()
