import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function to create a mask from sequence lengths.
def mask_from_lens(lens, max_len=None):
    """
    Creates a boolean mask from lengths.
    
    Args:
        lens (Tensor): a tensor of sequence lengths with shape [batch].
        max_len (int, optional): maximum length to generate the mask.
                                  If None, uses lens.max().
    Returns:
        Tensor: mask of shape [batch, max_len] (True for valid positions).
    """
    if max_len is None:
        max_len = lens.max().item()
    # Create a tensor of shape [max_len] and expand to [batch, max_len]
    ids = torch.arange(max_len, device=lens.device, dtype=lens.dtype)
    return ids.expand(lens.size(0), max_len) < lens.unsqueeze(1)

# -------------------- Positional Embedding --------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        # Compute inverse frequencies for sinusoid patterns.
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        """
        Args:
            pos_seq (Tensor): 1D tensor of positions (e.g., torch.arange(L)).
            bsz (int, optional): if provided, expands the positional embeddings to [bsz, L, demb].
        Returns:
            Tensor: positional embeddings of shape [1, L, demb] or [bsz, L, demb].
        """
        # Compute sinusoid input [L, num_inv_freq] then generate sin & cos embeddings.
        sinusoid_inp = torch.matmul(pos_seq.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        else:
            return pos_emb.unsqueeze(0)

# -------------------- Positionwise Convolutional Feed–Forward --------------------
class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.Conv1d(d_inner, d_model, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        # Transpose to apply Conv1d over the time dimension.
        if self.pre_lnorm:
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)
            output = core_out + inp  # Residual connection.
        else:
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)
            output = self.layer_norm(inp + core_out).to(inp.dtype)
        return output

# -------------------- Multi–Head Attention --------------------
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        # Linear projection for query, key and value.
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

        # Linear projection to get Q, K, V and reshape.
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
            # Repeat mask for each head.
            attn_mask = attn_mask.repeat(n_head, 1, 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        # Reshape attention output.
        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            output = residual + attn_out
        else:
            output = self.layer_norm(residual + attn_out)

        return output.to(attn_out.dtype)

# -------------------- Transformer Layer --------------------
class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, **kwargs):
        super(TransformerLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None):
        # Invert the mask for attention, then apply it.
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output = output * mask
        output = self.pos_ff(output)
        output = output * mask
        return output

# -------------------- FFTransformer --------------------
class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model, padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None, conditioning=0):
        # If no word embeddings are used, assume dec_inp is already a continuous representation.
        if self.word_emb is None:
            inp = dec_inp
            # Create a mask from the provided sequence lengths.
            mask = mask_from_lens(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        # Build positional embeddings using actual time dimension from inp.
        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask

        out = self.drop(inp + pos_emb + conditioning)

        for layer in self.layers:
            out = layer(out, mask=mask)

        return out, mask
