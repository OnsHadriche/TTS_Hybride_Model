#!/usr/bin/env python
"""
HybrideTTS Model with integrated Transformer, Attention, and Alignment modules.
This code integrates:
  - Transformer blocks (PositionalEmbedding, PositionwiseConvFF, MultiHeadAttn,
    TransformerLayer, FFTransformer) used in the decoder,
  - A convolutional attention block (ConvAttention, Invertible1x1ConvLUS, ConvNorm),
  - Alignment utilities (monotonic alignment search using Numba).
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from typing import Optional
# -------------------- Helper Functions --------------------
def get_mask_from_lengths(lengths, max_length=None):
    """
    Creates a boolean mask from sequence lengths.

    Args:
        lengths (Tensor): a tensor of sequence lengths, shape [batch].
        max_length (int, optional): If provided, uses this as the maximum length.
                                    Otherwise, uses lengths.max().
    Returns:
        Tensor: a mask of shape [batch, max_length] (True for positions within length).
    """
    if max_length is None:
        max_length = lengths.max().item()
    batch_size = lengths.size(0)
    ids = torch.arange(max_length, device=lengths.device)
    mask = ids.expand(batch_size, max_length) < lengths.unsqueeze(1)
    return mask
# ==================== Alignment Functions (Using Numba) ====================
from numba import jit, prange

@jit(nopython=True)
def mas(log_attn_map, width=1):
    # assumes mel x text; dynamic programming for monotonic alignment search
    opt = np.zeros_like(log_attn_map)
    log_attn_map = log_attn_map.copy()
    log_attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(log_attn_map)
    log_p[0, :] = log_attn_map[0, :]
    prev_ind = np.zeros_like(log_attn_map, dtype=np.int64)
    for i in range(1, log_attn_map.shape[0]):
        for j in range(log_attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j-width), j+1)
            prev_log = np.array([log_p[i-1, idx] for idx in prev_j])
            ind = np.argmax(prev_log)
            log_p[i, j] = log_attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]
    # now backtrack
    curr_text_idx = log_attn_map.shape[1]-1
    for i in range(log_attn_map.shape[0]-1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt

@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i-1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2
    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1]-1
    for i in range(log_p.shape[0]-1, 0, -1):
        opt[i, j] = one
        if log_p[i-1, j-1] >= log_p[i-1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt

@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1  # only supports width=1
    attn_out = np.zeros_like(b_log_attn_map)
    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, :out_lens[b], :in_lens[b]])
        attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
    return attn_out

# ==================== Transformer & Attention Modules ====================

# --- Convolution Normalization ---
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)

# --- Invertible 1x1 Convolution using LU factorization ---
class Invertible1x1ConvLUS(nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        W, _ = torch.linalg.qr(torch.randn(c, c))
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        p, lower, upper = torch.linalg.lu_factor(W)
        self.register_buffer('p', p)
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse[..., None]
            return F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W

# --- Positional Embedding ---
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(pos_seq.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        else:
            return pos_emb.unsqueeze(0)

# --- Positionwise Feed-Forward ---
class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()
        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.Conv1d(d_inner, d_model, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)
            output = core_out + inp
        else:
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)
            output = self.layer_norm(inp + core_out).to(inp.dtype)
        return output

# --- Multi-Head Attention ---
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        residual = inp
        if self.pre_lnorm:
            inp = self.layer_norm(inp)
        n_head, d_head = self.n_head, self.d_head
        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)
        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, 1, 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)
        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = residual + attn_out
        else:
            output = self.layer_norm(residual + attn_out)
        return output.to(attn_out.dtype)

# --- Transformer Layer ---
class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, **kwargs):
        super(TransformerLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout, pre_lnorm=kwargs.get('pre_lnorm', False))
    
    def forward(self, dec_inp, mask=None):
        # For attention mask, we invert the mask.
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output

# --- FFTransformer (Decoder) ---
class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.padding_idx = padding_idx
        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model, padding_idx=padding_idx)
        else:
            self.word_emb = None
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(n_head, d_model, d_head, d_inner, kernel_size, dropout,
                                 dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None, conditioning=0):
        if self.word_emb is None:
            inp = dec_inp
            mask = get_mask_from_lengths(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            mask = (dec_inp != self.padding_idx).unsqueeze(2)
        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb + conditioning)
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out, mask

# --- Temporal Predictor ---
class TemporalPredictor(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()
        layers = []
        for i in range(n_layers):
            layers += [
                ConvNorm(input_size if i == 0 else filter_size,
                         filter_size,
                         kernel_size=kernel_size,
                         stride=1,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(filter_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.layers = nn.Sequential(*layers)
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out

# --- Encoder with Multi-Speaker Support ---
class Encoder(nn.Module):
    def __init__(self, encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size,
                 num_speakers, speaker_embedding_dim):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim, encoder_embedding_dim,
                         kernel_size=encoder_kernel_size,
                         stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        self.gru = nn.GRU(encoder_embedding_dim + speaker_embedding_dim,
                          int(encoder_embedding_dim / 2),
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.5)

    @torch.jit.ignore
    def forward(self, x, input_lengths, speaker_ids):
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        if speaker_ids.dim() == 1:
            speaker_ids = speaker_ids.unsqueeze(1).expand(-1, x.size(1))
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        x = torch.cat([x, speaker_embeddings], dim=-1)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

# --- PostNet (Mel Refinement) ---
class PostNet(nn.Module):
    def __init__(self, n_mel, n_convolutions=5, channels=512, kernel_size=5, dropout=0.5):
        super(PostNet, self).__init__()
        convs = []
        for i in range(n_convolutions):
            in_ch = n_mel if i == 0 else channels
            out_ch = n_mel if i == n_convolutions - 1 else channels
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_ch),
                nn.Tanh() if i < n_convolutions - 1 else nn.Identity(),
                nn.Dropout(dropout)
            ]
        self.convs = nn.Sequential(*convs)

    def forward(self, mel):
        return self.convs(mel.transpose(1, 2)).transpose(1, 2)

# --- HybrideTTS Model ---
class HybrideTTS(nn.Module):
    def __init__(self,
                 n_symbols: int = 148,
                 symbols_embedding_dim: int = 512,
                 encoder_embedding_dim: int = 512,
                 encoder_n_convolutions: int = 3,
                 encoder_kernel_size: int = 5,
                 num_speakers: int = 40,
                 speaker_embedding_dim: int = 128,
                 dur_predictor_filter_size: int = 256,
                 dur_predictor_kernel_size: int = 3,
                 p_dur_predictor_dropout: float = 0.1,
                 out_fft_n_layers: int = 6,
                 out_fft_n_heads: int = 8,
                 out_fft_d_head: int = 128,
                 out_fft_conv1d_kernel_size: int = 3,
                 out_fft_conv1d_filter_size: int = 512,
                 out_fft_output_size: int = 512,
                 p_out_fft_dropout: float = 0.1,
                 p_out_fft_dropatt: float = 0.1,
                 p_out_fft_dropemb: float = 0.1,
                 n_mel_channels: int = 80,
                 pace: float = 1.0,
                 vocoder: nn.Module = None):
        super(HybrideTTS, self).__init__()

        self.n_mels = n_mel_channels
        self.pace = pace
        self.vocoder = vocoder
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = Encoder(encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size,
                               num_speakers, speaker_embedding_dim)
        self.duration_predictor = TemporalPredictor(
            input_size=encoder_embedding_dim,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout
        )
        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers,
            n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)
        self.postnet = PostNet(n_mel_channels)

    def forward(self, tokens: torch.Tensor, token_lengths: torch.Tensor,
                mel_targets: torch.Tensor, mel_lengths: torch.Tensor,
                speaker_ids: torch.Tensor):
        embedded = self.embedding(tokens).transpose(1, 2)
        enc_out = self.encoder(embedded, token_lengths, speaker_ids)
        enc_mask = get_mask_from_lengths(token_lengths).unsqueeze(-1).float()
        if enc_mask.size(1) < enc_out.size(1):
            enc_mask = F.pad(enc_mask, (0, enc_out.size(1) - enc_mask.size(1)))
        elif enc_mask.size(1) > enc_out.size(1):
            enc_mask = enc_mask[:, :enc_out.size(1)]
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1), min=1)
        
        # Use mel_targets.size(2) (fixed mel length) to regulate length.
        upsampled_enc, dec_lens = regulate_len(dur_pred, enc_out, self.pace, mel_targets.size(2))
        dec_out, dec_mask = self.decoder(upsampled_enc, mel_lengths)
        mel_out = self.proj(dec_out)
        mel_postnet_out = self.postnet(mel_out)
        return mel_out, dec_mask, dur_pred, log_dur_pred, dec_out, mel_postnet_out

    def infer(self, tokens, token_lens, speaker_ids):
        embedded = self.embedding(tokens).transpose(1, 2)
        enc_out = self.encoder(embedded, token_lens, speaker_ids)
        mask = get_mask_from_lengths(token_lens, enc_out.size(1)).unsqueeze(-1).float()
        log_dur = self.duration_predictor(enc_out, mask).squeeze(-1)
        dur = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
        enc_up, mel_lens = regulate_len(dur, enc_out, self.pace)
        dec_out, _ = self.decoder(enc_up, mel_lens)
        mel_out = self.proj(dec_out)
        mel_out = mel_out.permute(0, 2, 1)
        if self.vocoder is None:
            return mel_out, mel_lens, dur
        wav = self.vocoder(mel_out, speaker_ids)
        return wav, mel_out, mel_lens, dur

# --- Utility Functions ---
def regulate_len(durations, enc_out, pace: float = 1.0, mel_max_len: Optional[int] = None):
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)
    max_len = dec_lens.max().item()  # Use .item() for scalar
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)
    range_ = torch.arange(max_len, device=enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)  # [batch_size, max_len, embedding_dim]
    if mel_max_len is not None:
        batch_size, _, embedding_dim = enc_rep.size()
        if max_len < mel_max_len:
            # Pad with zeros if max_len < mel_max_len
            pad = torch.zeros(batch_size, mel_max_len - max_len, embedding_dim, 
                            device=enc_rep.device, dtype=enc_rep.dtype)
            enc_rep = torch.cat([enc_rep, pad], dim=1)
        elif max_len > mel_max_len:
            # Truncate if max_len > mel_max_len
            enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.full((batch_size,), mel_max_len, dtype=dec_lens.dtype, 
                            device=dec_lens.device)
    return enc_rep, dec_lens

def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))
    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)
    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()
    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg
