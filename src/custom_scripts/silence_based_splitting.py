import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import os
from pathlib import Path

def split_on_silence_with_length(audio_path, min_length=10, max_length=30, 
                               silence_thresh=-40, min_silence_len=500):
    """
    Split audio file into chunks based on silence and desired length constraints
    
    Parameters:
        audio_path: path to audio file
        min_length: minimum segment length in seconds
        max_length: maximum segment length in seconds
        silence_thresh: silence threshold in dB
        min_silence_len: minimum silence length in ms
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True
    )
    
    segments = []
    current_segment = chunks[0]
    
    # Combine or split chunks to meet length constraints
    for chunk in chunks[1:]:
        chunk_length = len(current_segment) / 1000  # Convert to seconds
        
        if chunk_length + len(chunk)/1000 <= max_length:
            current_segment += chunk
        else:
            if len(current_segment)/1000 >= min_length:
                segments.append(current_segment)
            current_segment = chunk
    
    # Add the last segment if it meets criteria
    if len(current_segment)/1000 >= min_length:
        segments.append(current_segment)
    
    return segments

if __name__ == "__main__":

    # Create output directory if it doesn't exist
    output_base_dir = Path("../../data/audio_split")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Get all mp3 files from input directory
    input_dir = Path("../../data/audio_trimmed")
    mp3_files = list(input_dir.glob("*.mp3"))

    for mp3_file in mp3_files:
        # Create output directory for this video
        video_name = mp3_file.stem
        output_dir = output_base_dir / video_name
        output_dir.mkdir(exist_ok=True)

        # Split audio into segments
        segments = split_on_silence_with_length(str(mp3_file))

        # Save each segment
        for i, segment in enumerate(segments):
            output_path = output_dir / f"{video_name}_segment_{i:03d}.wav"
            segment.export(str(output_path), format="wav")

        print(f"Processed {video_name}: Created {len(segments)} segments")
