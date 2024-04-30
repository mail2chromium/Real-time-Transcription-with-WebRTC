import soundfile as sf
import numpy as np
from whisper import load_model

# Load the audio file
audio_data, samplerate = sf.read('sample2.mp3')
print(f"Audio data shape: {audio_data.shape}")
print(f"Sample rate: {samplerate}")

# Ensure the audio data is float32, as expected by the Whisper model
audio_data = audio_data.astype(np.float32)

# Load the Whisper model
model = load_model("base")

# Optionally, check if audio needs resampling
if samplerate != 16000:
    import librosa
    audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
    samplerate = 16000

# Check the length of the audio data after resampling (if any)
print(f"Resampled audio data length: {len(audio_data)}")

# Run the transcription
result = model.transcribe(audio_data)

# Output the transcription result
print(result["text"])
