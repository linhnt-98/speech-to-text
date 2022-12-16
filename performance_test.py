from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer
import glob
import librosa
import numpy as np

labels = []
with open('\\70970\\61-70970.trans.txt') as f:
    for text_line in f:
        labels.append(text_line.split()[1])

model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')


def process_audio(tsr):
    inputs = tokenizer(tsr, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=-1)
    text = tokenizer.batch_decode(tokens)
    return str(text)


path = r'\\70970\\*.flac'
files = glob.glob(path)
transcripts = []
file_no = 0
for file in files:
    input_audio = librosa.load(file)  # paste file path here
    tensor = torch.tensor(np.array(input_audio[0]))
    transcripts.append(process_audio(tensor))
    file_no += 1
    print('file number ' + str(file_no) + ' finished')

print("WER:", wer(labels, transcripts))
