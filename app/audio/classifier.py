"""
Non-Linguistic Vocal Sound Filter (NSVF) - Layer 5.5
===================================================

Uses YAMNet (via TensorFlow Hub) to detect and reject non-speech vocal sounds:
- Coughs
- Sneezes
- Breathing / Panting
- Throat clearing
- Mic bumps / Impact sounds
- Dog barking/Animal sounds

Reference: https://tfhub.dev/google/yamnet/1
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import requests

# Class names mapping (YAMNet class map)
# We embed the critical ones or load from official CSV to ensure index accuracy
# YAMNet outputs 521 scores. We need to map indices to names.

class SoundEventClassifier:
    def __init__(self, device="cpu"):
        print("[Sound Classifier] Loading YAMNet model...")
        # Load YAMNet from TFHub
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("[Sound Classifier] YAMNet loaded successfully")
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Define Blocklist (These sounds terminate interruption)
        self.blocklist = {
            'Cough', 'Sneeze', 'Throat clearing', 'Breathing', 'Wheeze', 
            'Sniff', 'Gasp', 'Pant', 'Sigh', 'Groan', 'Grunt',
            'Burp, eructation', 'Hiccup', 
            'Finger snapping', 'Hands', 'Finger', # Mic handling noise
            'Knock', 'Tap', 'Clicking', 'Tick', # Mic bumps
            'Wind', 'Rustle', 'Static', 'White noise', 
            'Silence', 'Respiratory sounds',
            'Dog', 'Bark', 'Domestic animals, pets', 'Animal', 'Meow', 'Cat', 
            'Growling', 'Whimper (dog)'
        }
        
        print(f"[Sound Classifier] Blocklist configured with {len(self.blocklist)} sound types")

    def _load_class_names(self):
        """Load YAMNet class map"""
        try:
            class_map_csv_text = requests.get(
                'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
            ).text
            class_names = []
            reader = csv.reader(io.StringIO(class_map_csv_text))
            next(reader) # Skip header
            for row in reader:
                class_names.append(row[2])
            return np.array(class_names)
        except Exception as e:
            print(f"[Sound Classifier] Error loading class map: {e}")
            # Fallback for critical indices if offline (simplified)
            # This is risky, ideally we bundle the CSV.
            # Returning empty will cause logic to fail safe (allow speech)
            return np.array([])

    def classify(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[bool, str, float]:
        """
        Classify audio segment.
        
        Returns:
            (is_safe, label, confidence)
            is_safe: True if speech/neutral, False if blocked sound
        """
        try:
            if len(self.class_names) == 0:
                print("[Sound Classifier] Warning: Class map missing, passing audio")
                return True, "unknown", 0.0

            # YAMNet expects 16kHz float32 between -1.0 and +1.0
            # Ensure audio is properly normalized
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            # Run inference
            scores, embeddings, spectrogram = self.model(audio_data)
            
            # Scores is [N, 521] where N is number of 0.48s frames
            # We average the scores across frames for the whole segment
            avg_scores = np.mean(scores.numpy(), axis=0)
            
            # Get top prediction
            top_class_index = np.argmax(avg_scores)
            top_class = self.class_names[top_class_index]
            confidence = avg_scores[top_class_index]
            
            # Check blocklist
            if top_class in self.blocklist:
                return False, top_class, confidence
                
            # Special check: If "Speech" is NOT in top 3, be suspicious
            # But mostly we trust the top prediction vs blocklist
            
            return True, top_class, confidence
            
        except Exception as e:
            print(f"[Sound Classifier] Inference error: {e}")
            return True, "error", 0.0
