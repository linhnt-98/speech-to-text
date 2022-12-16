import streamlit as st
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import speech_recognition as sr
import io
import keyboard
from pydub import AudioSegment
import librosa
import numpy as np

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')

recognizer = sr.Recognizer()
is_session_in_progress = False

def process_audio(tsr):
    inputs = tokenizer(tsr, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=-1)
    text = tokenizer.batch_decode(tokens)
    return text[0].lower()


def transcribe_from_file(input_file):
    audio_data = librosa.load(input_file)
    tensor = torch.tensor(np.array(audio_data[0]))
    st.text_area('Transcript', process_audio(tensor))


def transcribe_from_voice(is_in_progress):
    with sr.Microphone(sample_rate=16000) as source:
        st.write("Start speaking!")
        while is_session_in_progress:
            audio = recognizer.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            clip = AudioSegment.from_wav(data)
            tensor = torch.tensor(clip.get_array_of_samples())

            st.write(process_audio(tensor))


st.title('Speech-to-Text Program')
st.write('Welcome! You can try transcribing an audio file, or live-transcribing your voice!')
st.header('Transcribe audio')
audio_input = st.file_uploader("Add your file here:")

if audio_input is not None:
    transcribe_from_file(audio_input)

st.header('Live-transcription')
container = st.empty()
start_button = container.button('Start')

if start_button:
    container.empty()
    stop_button = container.button('Stop')
    if is_session_in_progress is False:
        is_session_in_progress = True
        transcribe_from_voice(is_session_in_progress)
    else:
        is_session_in_progress = False




