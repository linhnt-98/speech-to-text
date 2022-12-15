import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import speech_recognition as sr
import io
import keyboard
from pydub import AudioSegment
import librosa
import numpy as np

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

recognizer = sr.Recognizer()


def process_audio(tsr):
    inputs = tokenizer(tsr, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=-1)
    text = tokenizer.batch_decode(tokens)
    return str(text)


"""
Menu
"""
print("Welcome to the program! Select what you'd want to transcribe:")
print("1 - An audio file")
print("2 - Recording")
selection = input()

if selection == str(1):
    """
    Transcribe text from audio file.
    """
    input_audio = librosa.load("D:\\Recording.flac")
    tensor = torch.tensor(np.array(input_audio[0]))
    print(process_audio(tensor))

elif selection == str(2):
    """
    Record voice and transcribe text.
    """
    with sr.Microphone(sample_rate=16000) as source:
        print("Try speaking! Press Esc to end the session.")
        is_session_in_progress = True
        while is_session_in_progress:
            if keyboard.read_key() == "esc":
                print("Session ended. Thank you for trying!")
                is_session_in_progress = False
                break
            audio = recognizer.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            clip = AudioSegment.from_wav(data)
            tensor = torch.tensor(clip.get_array_of_samples())

            print(process_audio(tensor))
