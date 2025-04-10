import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# Import your hybrid TTS model (adjust the import path)
from hybride_model import HybrideTTS

# ----------------------------
# Define a multi-speaker TTS Dataset
# ----------------------------
class TTSDataset(Dataset):
    def __init__(self, data_dir: str, num_speakers: int = 40):
        """
        This dataset should load your data.
        Here we create dummy data with random speaker IDs for demonstration.
        """
        self.samples = []
        # For demonstration, let's create 1000 dummy samples.
        for i in range(1000):
            sample = {
                # A dummy token sequence of length between 30 and 70.
                'tokens': torch.randint(1, 148, (np.random.randint(30, 70),)),
                # Use the actual length of tokens.
                'token_lengths': None,  # we'll compute length from tokens below
                # A dummy mel spectrogram with 80 channels and time length between 300 and 500.
                'mel': torch.randn(80, np.random.randint(300, 500)),
                # Use the actual mel time length.
                'mel_length': None,
                # Random speaker id between 0 and num_speakers-1.
                'speaker_id': torch.randint(0, num_speakers, (1,)).item()
            }
            # Set lengths from tensors
            sample['token_lengths'] = sample['tokens'].size(0)
            sample['mel_length'] = sample['mel'].size(1)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (sample['tokens'],
                sample['token_lengths'],
                sample['mel'],
                sample['mel_length'],
                sample['speaker_id'])


def collate_fn(batch):
    # Batch is a list of tuples: (tokens, token_length, mel, mel_length, speaker_id)
    batch_tokens = [item[0] for item in batch]
    batch_token_lengths = torch.LongTensor([item[1] for item in batch])
    batch_mels = [item[2] for item in batch]
    batch_mel_lengths = torch.LongTensor([item[3] for item in batch])
    batch_speaker_ids = torch.LongTensor([item[4] for item in batch])
    
    # Pad token sequences to the maximum length in the batch.
    max_token_len = max([t.size(0) for t in batch_tokens])
    padded_tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    for i, t in enumerate(batch_tokens):
        padded_tokens[i, :t.size(0)] = t

    # Pad mel spectrograms along time dimension.
    n_mels = batch_mels[0].size(0)
    max_mel_len = max([m.size(1) for m in batch_mels])
    padded_mels = torch.zeros(len(batch), n_mels, max_mel_len)
    for i, m in enumerate(batch_mels):
        padded_mels[i, :, :m.size(1)] = m

    return padded_tokens, batch_token_lengths, padded_mels, batch_mel_lengths, batch_speaker_ids


# ----------------------------
# Define loss functions
# ----------------------------
def mel_loss(predicted: torch.Tensor, target: torch.Tensor):
    """Mean Squared Error Loss for mel spectrograms."""
    return nn.MSELoss()(predicted, target)

def duration_loss(log_dur_pred: torch.Tensor, true_durations: torch.Tensor):
    """MSE Loss for log-durations. true_durations should be provided from your alignments."""
    return nn.MSELoss()(log_dur_pred, torch.log(true_durations.float() + 1))


# ----------------------------
# Training loop
# ----------------------------
def train(model: HybrideTTS, dataloader: DataLoader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        tokens, token_lengths, mel_targets, mel_lengths, speaker_ids = batch
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        mel_targets = mel_targets.to(device)
        mel_lengths = mel_lengths.to(device)
        speaker_ids = speaker_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass. Model returns tuple: (mel_out, dec_mask, dur_pred, log_dur_pred, dec_out)
        mel_out, dec_mask, dur_pred, log_dur_pred, dec_out = model(tokens, token_lengths, mel_targets, mel_lengths, speaker_ids)
        
        # Compute mel reconstruction loss.
        loss_mel = mel_loss(mel_out, mel_targets)
        
        # Optionally compute duration loss.
        # For demonstration, we use dummy true durations (replace with real alignment durations).
        true_durations = torch.ones_like(dur_pred).to(device) * 5  # dummy value; update accordingly.
        loss_dur = duration_loss(log_dur_pred, true_durations)
        
        # Total loss: weight your losses as needed.
        loss = loss_mel + loss_dur
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader.
    dataset = TTSDataset(data_dir="", num_speakers=40)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Instantiate the hybrid TTS model.
    model = HybrideTTS(
        n_mel_channels=80,
        n_symbols=148,
        padding_idx=0,
        symbols_embedding_dim=512,
        in_fft_n_layers=3,
        in_fft_n_heads=8,
        in_fft_d_head=128,
        in_fft_conv1d_kernel_size=5,
        in_fft_conv1d_filter_size=512,
        in_fft_output_size=512,
        p_in_fft_dropout=0.1,
        p_in_fft_dropatt=0.1,
        p_in_fft_dropemb=0.1,
        out_fft_n_layers=6,
        out_fft_n_heads=8,
        out_fft_d_head=128,
        out_fft_conv1d_kernel_size=3,
        out_fft_conv1d_filter_size=512,
        out_fft_output_size=512,
        p_out_fft_dropout=0.1,
        p_out_fft_dropatt=0.1,
        p_out_fft_dropemb=0.1,
        dur_predictor_kernel_size=3,
        dur_predictor_filter_size=256,
        p_dur_predictor_dropout=0.1,
        dur_predictor_n_layers=2,
        pitch_predictor_kernel_size=3,
        pitch_predictor_filter_size=256,
        p_pitch_predictor_dropout=0.1,
        pitch_predictor_n_layers=2,
        pitch_embedding_kernel_size=3,
        energy_conditioning=False,
        energy_predictor_kernel_size=3,
        energy_predictor_filter_size=256,
        p_energy_predictor_dropout=0.1,
        energy_predictor_n_layers=2,
        energy_embedding_kernel_size=3,
        n_speakers=40,
        speaker_emb_weight=1.0,
        pitch_conditioning_formants=1
    ).to(device)
    
    # Set up the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        # Optionally, save checkpoints:
        torch.save(model.state_dict(), f"hybride_tts_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
