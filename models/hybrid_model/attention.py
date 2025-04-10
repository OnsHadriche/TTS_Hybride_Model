# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# -------------------- Convolutional Normalization --------------------
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        # Auto-compute padding if not provided (requires odd kernel_size).
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        # Xavier initialization with gain calculated for given nonlinearity.
        nn.init.xavier_uniform_(self.conv.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)

# -------------------- Invertible 1x1 Convolution with LU factorization --------------------
class Invertible1x1ConvLUS(nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Initialize a random orthonormal matrix.
        W, _ = torch.linalg.qr(torch.randn(c, c))
        # Ensure positive determinant.
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        p, lower, upper = torch.linalg.lu_factor(W)
        # p: permutation matrix; lower and upper: LU factors.
        self.register_buffer('p', p)
        # Lower triangular factor (without diagonal, which is ones).
        lower = torch.tril(lower, -1)
        # Identity diagonal.
        lower_diag = torch.diag(torch.eye(c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        # Reconstruct U and L.
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            # Use the inverse of W for reverse operation.
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse[..., None]
            return F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            # Log-determinant computed only from the diagonal of U.
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W

# -------------------- Convolutional Attention Module --------------------
class ConvAttention(nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=80, temperature=1.0,
                 n_mel_convs=2, align_query_enc_type='3xconv',
                 use_query_proj=True):
        """
        Convolutional Attention for Flowtron-style TTS training.

        Args:
            n_mel_channels (int): Number of mel-spectrogram channels.
            n_speaker_dim (int): Speaker embedding dimension.
            n_text_channels (int): Dimension of text encoder outputs.
            n_att_channels (int): Dimension for attention projections.
            temperature (float): Temperature scaling factor.
            n_mel_convs (int): Number of mel convolutions (not used explicitly here).
            align_query_enc_type (str): Type of query encoder; one of {"inv_conv", "3xconv"}.
            use_query_proj (bool): Whether to project the query.
        """
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        # Scaling factor for attention.
        self.att_scaling_factor = np.sqrt(n_att_channels)
        self.softmax = nn.Softmax(dim=3)
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.use_query_proj = bool(use_query_proj)

        # Configure query projection according to type.
        if align_query_enc_type == "inv_conv":
            self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        elif align_query_enc_type == "3xconv":
            self.query_proj = nn.Sequential(
                ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3,
                         bias=True, w_init_gain='relu'),
                nn.ReLU(),
                ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1,
                         bias=True),
                nn.ReLU(),
                ConvNorm(n_mel_channels, n_att_channels, kernel_size=1,
                         bias=True))
        else:
            raise ValueError("Unknown query encoder type specified")

        # Key projection network.
        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2,
                     kernel_size=3, bias=True, w_init_gain='relu'),
            nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels,
                     kernel_size=1, bias=True))

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """
        Sorts, packs, runs through a recurrent model, then unpacks data.

        Args:
            sorted_idx (Tensor): Sorting indices.
            unsort_idx (Tensor): Inverse sorting indices.
            lens (Tensor): Sequence lengths (in descending order).
            padded_data (Tensor): Padded input sequences.
            recurrent_model (nn.Module): RNN (or similar) to process the data.
        Returns:
            Tensor: Hidden vectors in original order.
        """
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def encode_query(self, query, query_lens):
        """
        If needed, use an RNN to encode the query.
        (This function is included as a placeholder; be sure to set self.query_lstm if used.)
        """
        query = query.permute(2, 0, 1)  # (seq_len, batch, feature)
        lens, ids = torch.sort(query_lens, descending=True)
        original_ids = [0] * lens.size(0)
        for i in range(len(ids)):
            original_ids[ids[i]] = i
        query_encoded = self.run_padded_sequence(ids, original_ids, lens,
                                                 query, self.query_lstm)
        query_encoded = query_encoded.permute(1, 2, 0)
        return query_encoded

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None,
                keys_encoded=None, attn_prior=None):
        """
        Compute an attention matrix between queries and keys using a simplified isotropic Gaussian.

        Args:
            queries (Tensor): [B, C, T1] tensor (e.g., mel-spectrogram features).
            keys (Tensor): [B, C2, T2] tensor (e.g., text encoder outputs).
            query_lens (Tensor): Lengths for each query.
            mask (Tensor, optional): Mask over keys (shape: [B, T2]) where 0 indicates padding.
            key_lens (Tensor, optional): Lengths for keys.
            keys_encoded (Tensor, optional): Pre-encoded keys.
            attn_prior (Tensor, optional): Prior attention probabilities.
        Returns:
            Tuple[Tensor, Tensor]: Attention weights [B, 1, T1, T2] and log-attention probabilities.
        """
        # Project keys.
        keys_enc = self.key_proj(keys)  # (B, n_att_channels, T2)

        # Project queries (if enabled); otherwise use raw queries.
        if self.use_query_proj:
            if isinstance(self.query_proj, Invertible1x1ConvLUS):
                queries_enc, log_det_W = self.query_proj(queries)
            else:  # assume sequential for "3xconv"
                queries_enc = self.query_proj(queries)
                log_det_W = 0.0
        else:
            queries_enc, log_det_W = queries, 0.0

        # Compute squared differences.
        # Output shape: (B, n_att_channels, T1, T2)
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # Sum squared differences over the attention dimension and scale.
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
        attn_logprob = attn.clone()
        if mask is not None:
            # Expand mask and set padded positions to -infinity.
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2),
                                   -float("inf"))
        attn = self.softmax(attn)  # Softmax along T2 dimension.
        return attn, attn_logprob
