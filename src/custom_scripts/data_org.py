import pandas as pd
import os
import shutil
from tqdm import tqdm

def prepare_dataset():
    # Create necessary directories
    output_dir = '/workspace/F5-TTS/data/singapore_parliament'
    wavs_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)
    
    # Load the corrected info CSV
    df = pd.read_csv('/workspace/F5-TTS/data/audio_corrected_info.csv')
    
    # Filter high confidence entries
    high_conf_df = df[df['correction_confidence'] > 0.95].copy()
    
    # Prepare metadata
    metadata = []
    
    # Copy audio files and prepare metadata
    for _, row in tqdm(high_conf_df.iterrows(), desc="Copying files", total=len(high_conf_df)):
        # Get source audio path
        src_path = row['audio_path']
        if not os.path.isabs(src_path):
            src_path = os.path.join('/workspace/F5-TTS', src_path)
            
        # Keep original filename
        filename = os.path.basename(src_path)
        dest_path = os.path.join(wavs_dir, filename)
        
        # Copy the audio file
        try:
            shutil.copy2(src_path, dest_path)
            
            # Add to metadata
            metadata.append({
                'audio_file': f"wavs/{filename}",
                'text': row['corrected_text']
            })
        except FileNotFoundError:
            print(f"Warning: Could not find audio file: {src_path}")
            continue
        except Exception as e:
            print(f"Error processing {src_path}: {str(e)}")
            continue
    
    # Create metadata DataFrame and save
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"Dataset prepared successfully:")
    print(f"- Total files copied: {len(metadata)}")
    print(f"- Metadata saved to: {os.path.join(output_dir, 'metadata.csv')}")
    print(f"- Audio files saved to: {wavs_dir}")

if __name__ == "__main__":
    prepare_dataset()