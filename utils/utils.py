import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile  # for .flac support
import torch
import torch.distributed as dist
from scipy.io.wavfile import read


# Utility functions

def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    return torch.lt(ids, lens.unsqueeze(1))


def get_mask_from_lengths(lengths, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lengths.max()
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    return torch.le(mask, 0)

# def get_mask_from_lengths(lengths):
    # max_len = torch.max(lengths).item()
    # ids = torch.arange(0, max_len, device=lengths.device)
    # mask = (ids < lengths.unsqueeze(1)).bool()
    # return mask

def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def l2_promote():
    _libcudart = ctypes.CDLL('libcudart.so')
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)


def print_once(*msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*msg)


# Audio loading

def load_wav(full_path, torch_tensor=False):
    data, sampling_rate = soundfile.read(full_path, dtype='int16')
    if torch_tensor:
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    return data, sampling_rate


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# Dataset utilities

def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|"):
    def split_line(root, line):
        parts = line.strip().split(split)
        if has_speakers:
            paths, non_paths = parts[:-2], parts[-2:]
        else:
            paths, non_paths = parts[:-1], parts[-1:]
        return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            fpaths_and_text += [split_line(dataset_path, line) for line in f]
    return fpaths_and_text


# Config parsing
class ParseFromConfigFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with open(values, 'r') as f:
            data = json.load(f)
        for group in data:
            for k, v in data[group].items():
                setattr(namespace, k.replace('-', '_'), v)


# Checkpoint loading

def load_pretrained_weights(model, ckpt_fpath):
    model = getattr(model, "module", model)
    weights = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
    weights = {re.sub("^module.", "", k): v for k, v in weights.items()}

    ckpt_emb = weights["encoder.word_emb.weight"]
    new_emb = model.state_dict()["encoder.word_emb.weight"]

    ckpt_vocab_size = ckpt_emb.size(0)
    new_vocab_size = new_emb.size(0)
    if ckpt_vocab_size != new_vocab_size:
        print("WARNING: Resuming from a checkpoint with different vocab size.")
        min_len = min(ckpt_vocab_size, new_vocab_size)
        weights["encoder.word_emb.weight"][:min_len] = ckpt_emb[:min_len]

    model.load_state_dict(weights)


# Dict wrappers
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class DefaultAttrDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        return self[item]


# Benchmarking
class BenchmarkStats:
    def __init__(self):
        self.num_frames = []
        self.losses = []
        self.mel_losses = []
        self.took = []

    def update(self, num_frames, losses, mel_losses, took):
        self.num_frames.append(num_frames)
        self.losses.append(losses)
        self.mel_losses.append(mel_losses)
        self.took.append(took)

    def get(self, n_epochs):
        frames_s = sum(self.num_frames[-n_epochs:]) / sum(self.took[-n_epochs:])
        return {
            'frames/s': frames_s,
            'loss': np.mean(self.losses[-n_epochs:]),
            'mel_loss': np.mean(self.mel_losses[-n_epochs:]),
            'took': np.mean(self.took[-n_epochs:]),
            'benchmark_epochs_num': n_epochs
        }

    def __len__(self):
        return len(self.losses)


# Checkpoint manager
class Checkpointer:
    def __init__(self, save_dir, keep_milestones=[]):
        self.save_dir = save_dir
        self.keep_milestones = keep_milestones
        self.tracked = OrderedDict(sorted([
            (int(re.search("_(\d+).pt", fn).group(1)), fn)
            for fn in glob.glob(f"{save_dir}/FastPitch_checkpoint_*.pt")
        ], key=lambda t: t[0]))

    def last_checkpoint(self, output):
        def corrupted(fpath):
            try:
                torch.load(fpath, map_location="cpu")
                return False
            except:
                warnings.warn(f"Cannot load {fpath}")
                return True

        saved = sorted(
            glob.glob(f"{output}/FastPitch_checkpoint_*.pt"),
            key=lambda f: int(re.search("_(\d+).pt", f).group(1))
        )

        if saved and not corrupted(saved[-1]):
            return saved[-1]
        elif len(saved) >= 2:
            return saved[-2]
        return None

    def maybe_load(self, model, optimizer, scaler, train_state, args, ema_model=None):
        assert args.checkpoint_path is None or not args.resume, (
            "Specify only one checkpoint source.")

        fpath = args.checkpoint_path if args.checkpoint_path else (
            self.last_checkpoint(args.output) if args.resume else None)

        if fpath is None:
            return

        print_once(f"Loading model and optimizer from {fpath}")
        ckpt = torch.load(fpath, map_location="cpu")
        train_state.update(epoch=ckpt["epoch"] + 1, total_iter=ckpt["iteration"])

        def no_pref(sd):
            return {re.sub("^module.", "", k): v for k, v in sd.items()}

        model.load_state_dict(no_pref(ckpt["state_dict"]))
        if ema_model is not None:
            ema_model.load_state_dict(no_pref(ckpt["ema_state_dict"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        else:
            warnings.warn("AMP scaler state missing.")

    def maybe_save(self, args, model, ema_model, optimizer, scaler, epoch, total_iter, config):
        if not (epoch == args.epochs or epoch % args.epochs_per_checkpoint == 0 or epoch in self.keep_milestones):
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        ckpt = {
            "epoch": epoch,
            "iteration": total_iter,
            "config": config,
            "train_setup": args.__dict__,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }
        if ema_model is not None:
            ckpt["ema_state_dict"] = ema_model.state_dict()

        fpath = Path(args.output, f"FastPitch_checkpoint_{epoch}.pt")
        print(f"Saving checkpoint to {fpath}")
        torch.save(ckpt, fpath)

        self.tracked[epoch] = fpath
        for e in set(list(self.tracked)[:-2]) - set(self.keep_milestones):
            try:
                os.remove(self.tracked[e])
            except:
                pass
            del self.tracked[e]