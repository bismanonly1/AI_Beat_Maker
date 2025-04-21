import optuna
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
import torchaudio
import os

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
os.makedirs("genereated", exist_ok = True)

def generate(prompt, tempo):
    prompt = f"{prompt}, {tempo} BPM"
    inputs = processor(text = [prompt], padding = True, return_tensors = "pt")
    output = model.generate(**inputs, max_new_tokens = 300)
    waveform = processor.audio_to_waveform(output[0], sampling_rate = 16000)
    path = f"generated/{prompt.replace(' ', '_')}.wav"
    torchaudio.save(path, waveform, 16000)
    return path

def objective(trial):
    tempo = trial.suggest_int("tempo", 60, 160)
    mood = trial.suggest_categorial("mood", ["chill", "dark", "energetic"])
    path = generate(f"{mood} techno beat", tempo)
    return tempo

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 5)
print("Best parameters:", study.best_params)