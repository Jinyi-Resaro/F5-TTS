from transformers import pipeline
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import torch
import logging

# Configure logging
logging.basicConfig(
    filename='transcription.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


def transcribe_segments(audio_segments):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", 
                         model="openai/whisper-large-v3",
                         device=device)
    
    dataset = []
    durations = []
    
    for i, audio_path in enumerate(audio_segments):
        try:
            # Get transcription
            result = transcriber(audio_path)
            text = result["text"]
            
            # Get duration
            audio, sr = librosa.load(audio_path)
            duration = len(audio) / sr
            
            dataset.append({
                "audio_path": audio_path,
                "text": text,
                "duration": duration
            })
            durations.append(duration)
            logging.info(f"Processed {i}th audio.")
        except:
            logging.info(f"{audio_path} processed failed")
            continue
    
    return dataset, durations

if __name__ == "__main__":

    # Get all audio files recursively under audio_split directory
    audio_split_dir = Path("/workspace/F5-TTS/data/audio_split")
    audio_files = []
    
    for ext in ['*.mp3', '*.wav']:
        audio_files.extend(list(audio_split_dir.rglob(ext)))
    
    logging.info(f"Found {len(audio_files)} audio files")
    
    # Convert paths to strings
    audio_files = [str(p) for p in audio_files]
    
    # Transcribe all segments
    dataset, durations = transcribe_segments(audio_files)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(dataset)
    output_path = "/workspace/F5-TTS/data/audio_split_info.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Saved transcription info to {output_path}")
