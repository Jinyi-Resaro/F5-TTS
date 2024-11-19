from difflib import SequenceMatcher
from typing import List, Tuple
import numpy as np
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import os
from langdetect import detect

class TranscriptCorrector:
    def __init__(self, use_transformer: bool = False):
        """Initialize the transcript corrector.
        
        Args:
            use_transformer: Whether to use transformer embeddings for similarity matching.
                           Falls back to fuzzy string matching if False.
        """
        self.use_transformer = use_transformer
        if use_transformer:
            # Load a language model for better semantic matching
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').cuda()
    
    def get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get sentence embeddings using the transformer model."""
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = self.model(**tokens)
        # Use mean pooling to get sentence embeddings
        return torch.mean(outputs.last_hidden_state, dim=1)
    
    def find_best_match(self, auto_sentence: str, human_transcript: str, 
                   window_size: int = 100) -> Tuple[str, float]:
        """Find the best matching segment in the human transcript for an auto-generated sentence.
        
        Args:
            auto_sentence: The auto-generated sentence to correct
            human_transcript: The complete human-made transcript
            window_size: Initial window size for rough localization
        """
        if len(auto_sentence.strip()) == 0:
            return "", 0.0
            
        # First pass: Find approximate location using windows
        words = human_transcript.split()
        windows = []
        window_starts = []  # Track starting indices of windows
        for i in range(0, len(words), window_size // 2):
            window = " ".join(words[i:i + window_size])
            windows.append(window)
            window_starts.append(i)
            
        # Find best matching window
        if self.use_transformer:
            all_texts = [auto_sentence] + windows
            embeddings = self.get_embeddings(all_texts)
            auto_embedding = embeddings[0]
            window_embeddings = embeddings[1:]
            similarities = torch.nn.functional.cosine_similarity(
                auto_embedding.unsqueeze(0), 
                window_embeddings
            )
            best_window_idx = similarities.argmax().item()
            rough_score = similarities[best_window_idx].item()
        else:
            best_window_idx = 0
            rough_score = 0
            for idx, window in enumerate(windows):
                score = fuzz.ratio(auto_sentence.lower(), window.lower()) / 100
                if score > rough_score:
                    rough_score = score
                    best_window_idx = idx

        # Second pass: Fine-tune boundaries around best window
        start_idx = max(0, window_starts[best_window_idx] - window_size//2)
        end_idx = min(len(words), start_idx + window_size*2)
        search_region = words[start_idx:end_idx]
        
        # Convert auto_sentence to words for matching
        auto_words = auto_sentence.split()
        auto_len = len(auto_words)
        
        best_score = 0
        best_start = 0
        best_end = 0
        
        # Slide through search region with variable-length windows
        for i in range(len(search_region) - len(auto_words) + 1):
            # Try different window lengths around the approximate length
            for window_len in range(max(auto_len-3, 1), auto_len+4):
                if i + window_len > len(search_region):
                    continue
                    
                window = " ".join(search_region[i:i+window_len])
                
                # Calculate similarity score
                if self.use_transformer:
                    window_embedding = self.get_embeddings([window])[0]
                    score = torch.nn.functional.cosine_similarity(
                        auto_embedding.unsqueeze(0),
                        window_embedding.unsqueeze(0)
                    ).item()
                else:
                    score = fuzz.ratio(auto_sentence.lower(), window.lower()) / 100
                    
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = i + window_len
        
        best_match = " ".join(search_region[best_start:best_end])
        return best_match, best_score
    
    def correct_transcripts(self, 
                          auto_transcripts: List[str], 
                          human_transcript: str,
                          confidence_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Correct a list of auto-generated transcripts using the human transcript.
        
        Args:
            auto_transcripts: List of auto-generated transcripts
            human_transcript: The complete human-made transcript
            confidence_threshold: Minimum confidence score to accept a match
            
        Returns:
            List of tuples containing (original_text, corrected_text, confidence_score)
        """
        corrections = []
        
        for auto_text in auto_transcripts:
            best_match, confidence = self.find_best_match(auto_text, human_transcript)
            
            if confidence >= confidence_threshold:
                corrections.append((auto_text, best_match, confidence))
            else:
                corrections.append((auto_text, auto_text, confidence))  # Keep original if no good match
            # print(f'auto: {auto_text}.\ntranscribed: {best_match}\nconfidence: {confidence}')
                
        return corrections

def main():
    
    # Initialize transcript corrector
    corrector = TranscriptCorrector()
    
    # Read the original CSV
    df = pd.read_csv('/workspace/F5-TTS/data/audio_split_info.csv')
    
    # Get all txt files from hansard scripts directory
    hansard_dir = '/workspace/F5-TTS/data/hansard_scripts/'
    txt_files = [f for f in os.listdir(hansard_dir) if f.endswith('.txt')]
    
    # Create a copy of the dataframe for corrections
    corrected_df = df.copy()
    corrected_df['corrected_text'] = df['text']  # Initialize with original text
    corrected_df['correction_confidence'] = 0.0
    
    # Process each date's transcript
    for txt_file in txt_files:
        print(f'processing {txt_file}')
        date_str = txt_file.replace('.txt', '')
        
        # Filter rows for current date
        date_df = df[df['audio_path'].str.contains(date_str, na=False)]
        
        if len(date_df) == 0:
            continue
            
        # Load human transcript
        with open(os.path.join(hansard_dir, txt_file), 'r') as f:
            human_transcript = f.read()
        
        # Get auto transcripts for this date
        auto_transcripts = []
        indices = []
        
        # Filter for English texts
        for idx, row in tqdm(date_df.iterrows()):
            try:
                if detect(row['text']) == 'en':
                    auto_transcripts.append(row['text'])
                    indices.append(idx)
            except:
                continue  # Skip if language detection fails
        
        # Get corrections
        if auto_transcripts:
            corrections = corrector.correct_transcripts(auto_transcripts, human_transcript)
            
            # Update the corrected dataframe
            for (orig, corrected, conf), idx in zip(corrections, indices):
                corrected_df.loc[idx, 'corrected_text'] = corrected
                corrected_df.loc[idx, 'correction_confidence'] = conf
    
    # Save the corrected dataframe
    corrected_df.to_csv('/workspace/F5-TTS/data/audio_corrected_info.csv', index=False)
    print("Corrections saved to audio_corrected_info.csv")

if __name__ == "__main__":
    main()
