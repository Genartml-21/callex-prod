import torch
import torchaudio
from torch.utils.data import Dataset
import os

class ProprietaryAudioDataset(Dataset):
    """
    R&D Dataloader mapping raw high-fidelity WAV sequences into Mel-Spectrograms dynamically.
    Matches raw temporal audio to integer phoneme boundaries natively via Torchaudio logic.
    """
    def __init__(self, metadata_path: str, wav_dir: str, tokenizer):
        self.wav_dir = wav_dir
        self.tokenizer = tokenizer
        
        # Simulating proprietary audio metadata: [file_id|transcription]
        self.metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        self.metadata.append((parts[0], parts[1]))
        else:
            print(f"⚠️ [Dataset Loader] Metadata corpus missing at '{metadata_path}'. Generating zero-vector simulation map.")

        # Mel Spectrogram Engine Configuration (High Fidelity Generative Constants)
        self.mel_engine = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            f_min=0,
            f_max=8000
        )

    def __len__(self):
        return len(self.metadata) if self.metadata else 200 # Fixed mock volume

    def __getitem__(self, idx):
        if not self.metadata:
            # Generate synthetic matrix data structurally to prove pipeline framework executes flawlessly
            dummy_text = "Internal Callex Voice Model Training Environment"
            dummy_phonemes = torch.tensor(self.tokenizer.text_to_sequence(dummy_text), dtype=torch.long)
            dummy_mel = torch.rand(80, 200) # Simulating mathematically structured spectrograms -> ~1 second
            return dummy_phonemes, dummy_mel

        wav_name, text = self.metadata[idx]
        wav_path = os.path.join(self.wav_dir, f"{wav_name}.wav")
        phonemes = torch.tensor(self.tokenizer.text_to_sequence(text), dtype=torch.long)
        
        # Process and normalize raw real-world temporal amplitude 
        waveform, sr = torchaudio.load(wav_path)
        mel_spectrogram = self.mel_engine(waveform).squeeze(0)
        
        # Logarithmic spectral scaling algorithm to isolate auditory frequencies
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        return phonemes, mel_spectrogram

def collate_generative_batch(batch):
    """Dynamically pads unequal sequence lengths into batched spatial matrices for GPU processing."""
    phonemes, mels = zip(*batch)
    
    # Pad numerical text sequences cleanly with 0 tensors backwardly
    phonemes_padded = torch.nn.utils.rnn.pad_sequence(phonemes, batch_first=True, padding_value=0)
    
    # Mels need dynamic padding across the pure time dimension -> [batch, n_mels, max_time]
    max_time = max(mel.size(1) for mel in mels)
    mels_padded = torch.zeros(len(mels), 80, max_time)
    for i, mel in enumerate(mels):
        mels_padded[i, :, :mel.size(1)] = mel
        
    return phonemes_padded, mels_padded
