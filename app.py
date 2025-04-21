from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import torch
import torchaudio
import os
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

app = FastAPI()

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

os.makedirs("generated", exist_ok=True)

def generate_beat(genre, duration):
    prompt = f"a {genre} beat"
    inputs = processor(text=[prompt], padding = True, return_tensors = "pt")
    audio_values = model.generate(**inputs, max_new_tokens = duration * 50)
    output_path = f"generated/{genre}_{duration}s.wav"
    torchaudio.save(output_path, audio_values, 16000)
    return output_path

@app.post("/generate")
def generate_music(genre: str = Form(...), duration: int = Form(...)):
    file_path = generate_beat(genre, duration)
    return FileResponse(path = file_path, filename = "generated_beat.wav", media_type = "audio/wav")