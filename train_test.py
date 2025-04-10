import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import librosa

from models.hybrid_model.hybride_model import HybrideTTS

# Corrected collate function
def text_mel_collate_fn(batch):
    tokens, input_lengths, mels, gates, output_lengths, speaker_ids = zip(*batch)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([x.item() for x in input_lengths], dtype=torch.long)
    output_lengths = torch.tensor([x.item() for x in output_lengths], dtype=torch.long)
    mels = [m.T for m in mels]
    mel_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=-11.5129)
    mel_padded = mel_padded.transpose(1, 2)
    gate_padded = torch.nn.utils.rnn.pad_sequence(gates, batch_first=True, padding_value=1)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    return tokens_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids

# Function to extract unique speaker IDs from multiple dataset files
def extract_unique_speaker_ids(dataset_files):
    unique_speaker_ids = set()
    for dataset_file in dataset_files:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('" "')
                if len(parts) == 3:
                    speaker_id = parts[1].strip('"')
                    unique_speaker_ids.add(speaker_id)
    return list(unique_speaker_ids)

# Function to create a mapping from speaker ID strings to integers
def create_speaker_id_map(unique_speaker_ids):
    return {sid: idx for idx, sid in enumerate(unique_speaker_ids)}

# Custom Dataset class
class ArabDataset(Dataset):
    def __init__(self, dataset_file, speaker_id_map, audio_base_path=None):
        self.data = []
        self.speaker_id_map = speaker_id_map
        self.audio_base_path = audio_base_path or ""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('" "')
                if len(parts) == 3:
                    audio_path = parts[0].strip('"')
                    speaker_id_str = parts[1].strip('"')
                    transcription = parts[2].strip('"')
                    speaker_id_int = self.speaker_id_map[speaker_id_str]
                    full_audio_path = os.path.join(self.audio_base_path, audio_path)
                    self.data.append((full_audio_path, speaker_id_int, transcription))

    def __getitem__(self, idx):
        audio_path, speaker_id, transcription = self.data[idx]
        audio, sr = librosa.load(audio_path, sr=22050)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, hop_length=256, n_fft=1024
        )
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        tokens = [ord(c) % 40 for c in transcription[:20]]
        tokens = torch.tensor(tokens, dtype=torch.long)
        input_lengths = torch.tensor([len(tokens)], dtype=torch.long)
        output_lengths = torch.tensor([mel_spectrogram.size(1)], dtype=torch.long)
        gate = torch.zeros(mel_spectrogram.size(1))
        gate[output_lengths.item() - 1:] = 1
        return tokens, input_lengths, mel_spectrogram, gate, output_lengths, speaker_id

    def __len__(self):
        return len(self.data)

# Configuration class
class Config:
    def __init__(self):
        self.train_labels = os.path.join('data', 'custom_data', 'ready_train.txt')
        self.test_labels = os.path.join('data', 'custom_data', 'ready_val.txt')
        self.audio_base_path = os.path.join('data', 'custom_data', 'enhanced_audio')
        self.checkpoint_dir = os.path.join('data', 'checkpoints')
        self.log_dir = 'logs'
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.n_save_states_iter = 1000
        self.random_seed = 42

config = Config()

# Set random seed
if config.random_seed:
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create checkpoint directory
os.makedirs(config.checkpoint_dir, exist_ok=True)

# Extract speaker IDs from both train and test sets
unique_speaker_ids = extract_unique_speaker_ids([config.train_labels, config.test_labels])
speaker_id_map = create_speaker_id_map(unique_speaker_ids)

# Instantiate datasets
train_dataset = ArabDataset(config.train_labels, speaker_id_map, config.audio_base_path)
test_dataset = ArabDataset(config.test_labels, speaker_id_map, config.audio_base_path)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=text_mel_collate_fn,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=text_mel_collate_fn,
                         shuffle=False, drop_last=False)

# Model setup
num_speakers = len(speaker_id_map)
model = HybrideTTS(n_symbols=40, num_speakers=num_speakers, n_mel_channels=80).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Training loop
def train_model(model, optimizer, train_loader, test_loader, config, device):
    model.train()
    n_iter = 0
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch}")
        for batch in tqdm(train_loader):
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids = batch
            text_padded = text_padded.to(device)
            input_lengths = input_lengths.to(device)
            mel_padded = mel_padded.to(device)
            gate_padded = gate_padded.to(device)
            output_lengths = output_lengths.to(device)
            speaker_ids = speaker_ids.to(device)

            y_pred = model(text_padded, input_lengths, mel_padded, output_lengths, speaker_ids)
            mel_out, dec_mask, dur_pred, log_dur_pred, dec_out, mel_postnet_out = y_pred

            mel_loss = F.mse_loss(mel_out.transpose(1, 2), mel_padded) + F.mse_loss(mel_postnet_out.transpose(1, 2), mel_padded)
            loss = mel_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)
            optimizer.step()

            if n_iter % 100 == 0:
                print(f"Iteration: {n_iter}, Loss: {loss.item()}, Grad Norm: {grad_norm.item()}")

            if n_iter % config.n_save_states_iter == 0 and n_iter > 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f'states_{n_iter}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            n_iter += 1

        validate(model, test_loader, device)

# Validation function
@torch.inference_mode()
def validate(model, test_loader, device):
    model.eval()
    loss_sum = 0
    n_test_sum = 0
    for batch in test_loader:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids = batch
        text_padded = text_padded.to(device)
        input_lengths = input_lengths.to(device)
        mel_padded = mel_padded.to(device)
        gate_padded = gate_padded.to(device)
        output_lengths = output_lengths.to(device)
        speaker_ids = speaker_ids.to(device)

        y_pred = model(text_padded, input_lengths, mel_padded, output_lengths, speaker_ids)
        mel_out, dec_mask, dur_pred, log_dur_pred, dec_out, mel_postnet_out = y_pred

        mel_loss = F.mse_loss(mel_out, mel_padded) + F.mse_loss(mel_postnet_out, mel_padded)
        loss = mel_loss

        loss_sum += mel_padded.size(0) * loss.item()
        n_test_sum += mel_padded.size(0)

    val_loss = loss_sum / n_test_sum
    print(f"Validation Loss: {val_loss}")
    model.train()

if __name__ == "__main__":
    print(f"Training on {device} with {num_speakers} speakers")
    train_model(model, optimizer, train_loader, test_loader, config, device)