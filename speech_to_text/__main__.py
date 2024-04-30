import asyncio
import sys, os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

from faster_whisper import WhisperModel
from .audio_transcriber import AppOptions, AudioTranscriber
from .utils import audio_utils, file_utils
from .websoket_server import WebSocketServer
from .openai_api import OpenAIAPI
from .settings_manager import load_settings_from_file, filter_transcribe_settings

# Load and filter settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(BASE_DIR, "user_settings.json")
settings = load_settings_from_file(settings_path)

app_settings = settings['app_settings']
model_settings = settings['model_settings']
transcribe_settings = filter_transcribe_settings(settings['transcribe_settings'])

# Initialize components
event_loop = asyncio.new_event_loop()
transcriber = None
thread = None
websocket_server = WebSocketServer(event_loop) if app_settings['use_websocket_server'] else None
openai_api = OpenAIAPI() if app_settings['use_openai_api'] else None

# transcriber: AudioTranscriber = None
# event_loop: asyncio.AbstractEventLoop = None
# thread: threading.Thread = None
# websocket_server: WebSocketServer = None
# openai_api: OpenAIAPI = None

def load_audio_file(file_path):
    """Function to load and process audio file."""
    try:
        data, samplerate = sf.read(file_path)
        # Ensure sample rate compatibility
        if samplerate != 16000:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
        return data.astype(np.float32), 16000
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def record_audio(duration=10, samplerate=16000):
    """Function to record audio from the default microphone."""
    print(f"Recording {duration} seconds of audio...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()

def start_transcription(audio_data):
    """Start transcription using the Whisper model."""
    print("Starting transcription...")
    whisper_model = WhisperModel(model_settings['model_size_or_path'])
    global transcriber
    transcriber = AudioTranscriber(event_loop, whisper_model, transcribe_settings, AppOptions(**app_settings), websocket_server, openai_api)
    transcriber.batch_transcribe_audio(audio_data)

def main():
    if len(sys.argv) > 1:
        # Load audio from file
        audio_file_path = sys.argv[1]
        audio_data, rate = load_audio_file(audio_file_path)
        if audio_data is not None:
            start_transcription(audio_data)
        else:
            print("Failed to load audio data.")
    else:
        # Record audio from microphone
        audio_data = record_audio()
        start_transcription(audio_data)

if __name__ == "__main__":
    main()
