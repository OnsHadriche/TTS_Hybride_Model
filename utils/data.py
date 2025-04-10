import os
import re

# import text
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from utils import read_lines_from_file, progbar
from utils.audio import MelSpectrogram

def text_mel_collate_fn(batch):
    # Unpack the batch
    tokens, input_lengths, mels, gates, output_lengths, speaker_ids = zip(*batch)

    # Pad sequences
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    input_lengths = torch.stack(input_lengths)  # Stack into a tensor [batch_size]
    mel_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=-11.5129)  # Log-mel silence value
    gate_padded = torch.nn.utils.rnn.pad_sequence(gates, batch_first=True, padding_value=1)
    output_lengths = torch.stack(output_lengths)  # Stack into a tensor [batch_size]
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)  # Convert to tensor [batch_size]

    return tokens_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids


def normalize_pitch(pitch, 
                    mean: float = 130.05478, 
                    std: float = 22.86267):
    zeros = (pitch == 0.0)
    pitch -= mean
    pitch /= std
    pitch[zeros] = 0.0
    return pitch

def remove_silence(energy_per_frame: torch.Tensor, 
                   thresh: float = -10.0):
    keep = energy_per_frame > thresh
    # keep silence at the end
    i = keep.size(0)-1
    while not keep[i] and i > 0:
        keep[i] = True
        i -= 1
    return keep

def make_dataset_from_subdirs(folder_path):
    samples = []
    for root, _, fnames in os.walk(folder_path, followlinks=True):
        for fname in fnames:
            if fname.endswith('.wav'):
                samples.append(os.path.join(root, fname))

    return samples
    return phonemes, filename
    

class ArabDataset4FastPitch(Dataset):
    def __init__(self, 
                 txtpath: str = '',
                 wavpath: str = '',                
                 label_pattern: str = '"(?P<filename>.*)" "(?P<phonemes>.*)"',
                 f0_dict_path: str = 'data\custom_data\pitch_dict.pt',
                 f0_mean: float = 130.05478, 
                 f0_std: float = 22.86267,
                 sr_target: int = 22050
                 ):
        super().__init__()
        from models.hybrid_model.data_function import BetaBinomialInterpolator

        self.mel_fn = MelSpectrogram()
        self.wav_path = wavpath
        self.label_pattern = label_pattern
        self.sr_target = sr_target

        self.f0_dict = torch.load(f0_dict_path)
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.betabinomial_interpolator = BetaBinomialInterpolator()

        self.data = self._process_textfile(txtpath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        phonemes, fpath, pitch_mel = self.data[idx]

        wave, sr = torchaudio.load(fpath)
        if sr != self.sr_target:
            wave = torchaudio.functional.resample(wave, sr, self.sr_target, 64)

        mel_raw = self.mel_fn(wave)
        mel_log = mel_raw.clamp_min(1e-5).log().squeeze()

        keep = remove_silence(mel_log.mean(0))
        mel_log = mel_log[:, keep]
        pitch_mel = normalize_pitch(pitch_mel[:,keep], self.f0_mean, self.f0_std)

        energy = torch.norm(mel_log.float(), dim=0, p=2)
        attn_prior = torch.from_numpy(
            self.betabinomial_interpolator(mel_log.size(1), len(phonemes)))

        speaker = None
        return (phonemes, mel_log, len(phonemes), pitch_mel, 
                energy, speaker, attn_prior,
                fpath)
    

class DynBatchDataset(ArabDataset4FastPitch):
    def __init__(self, 
                 txtpath: str = 'data\custom_data/ready_train.txt',
                 wavpath: str = 'data/wav_new',
                 label_pattern: str = '"(?P<filename>.*)" "(?P<phonemes>.*)"',
                 f0_dict_path: str = 'data\custom_data\pitch_dict.pt',
                 f0_mean: float = 130.05478, 
                 f0_std: float = 22.86267,
                 max_lengths: list[int] = [1000, 1300, 1850, 30000],
                 batch_sizes: list[int] = [10, 8, 6, 4],
                 ):
        
        super().__init__(txtpath=txtpath, wavpath=wavpath,
                         label_pattern=label_pattern,
                         f0_dict_path=f0_dict_path,
                         f0_mean=f0_mean, f0_std=f0_std)

        self.max_lens = [0,] + max_lengths
        self.b_sizes = batch_sizes

        self.id_batches = []
        self.shuffle()

    def shuffle(self):
      
        lens = [x[2].size(1) for x in self.data] # x[2]: pitch

        ids_per_bs = {b: [] for b in self.b_sizes}

        for i, mel_len in enumerate(lens):
            b_idx = next(i for i in range(len(self.max_lens)-1)
                         if self.max_lens[i] <= mel_len < self.max_lens[i+1])
            ids_per_bs[self.b_sizes[b_idx]].append(i)

        id_batches = []

        for bs, ids in ids_per_bs.items():
            np.random.shuffle(ids)
            ids_chnk = [ids[i:i+bs] for i in range(0, len(ids), bs)]
            id_batches += ids_chnk

        self.id_batches = id_batches

    def __len__(self):
        return len(self.id_batches)

    def __getitem__(self, idx):
        batch = [super(DynBatchDataset, self).__getitem__(idx)
                 for idx in self.id_batches[idx]]
        return batch