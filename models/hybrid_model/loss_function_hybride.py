import torch
import torch.nn.functional as F
from torch import nn
from .attn_loss_function import AttentionCTCLoss  # assure-toi que ce module est bien importé

def mask_from_lens(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device)
    return ids.unsqueeze(0) < lens.unsqueeze(1)

class HybridTTSLoss(nn.Module):
    def __init__(self,
                 mel_loss_scale=1.0,
                 postnet_loss_scale=1.0,
                 gate_loss_scale=1.0,
                 dur_loss_scale=1.0,
                 pitch_loss_scale=1.0,
                 energy_loss_scale=0.1,
                 attn_loss_scale=1.0):
        super().__init__()
        self.mel_loss_scale = mel_loss_scale
        self.postnet_loss_scale = postnet_loss_scale
        self.gate_loss_scale = gate_loss_scale
        self.dur_loss_scale = dur_loss_scale
        self.pitch_loss_scale = pitch_loss_scale
        self.energy_loss_scale = energy_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, model_out, targets, epoch=None, max_epoch=None):
        # Unpack outputs
        (mel_out, mel_out_postnet, gate_out, dec_mask,
         dur_pred, log_dur_pred, pitch_pred, pitch_tgt,
         energy_pred, energy_tgt, attn_soft, attn_hard,
         attn_dur, attn_logprob) = model_out

        # Unpack targets
        mel_tgt, in_lens, out_lens, gate_tgt = targets

        # ===== Masking =====
        dur_mask = mask_from_lens(in_lens, dur_pred.size(1))  # (B, T_text)
        mel_mask = mask_from_lens(out_lens, mel_tgt.size(2))  # (B, T_mel)

        # ===== Duration loss =====
        log_dur_tgt = torch.log(attn_dur.float() + 1)
        dur_pred_loss = F.mse_loss(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        # ===== Mel loss =====
        mel_out = F.pad(mel_out, (0, 0, 0, mel_tgt.size(2) - mel_out.size(2)), value=0.0)
        mel_loss = F.mse_loss(mel_out, mel_tgt.transpose(1, 2), reduction='none')
        mel_loss = (mel_loss * mel_mask.unsqueeze(1).float()).sum() / mel_mask.sum()

        # ===== Postnet Mel loss =====
        mel_out_postnet = F.pad(mel_out_postnet, (0, 0, 0, mel_tgt.size(2) - mel_out_postnet.size(2)), value=0.0)
        postnet_loss = F.mse_loss(mel_out_postnet, mel_tgt.transpose(1, 2), reduction='none')
        postnet_loss = (postnet_loss * mel_mask.unsqueeze(1).float()).sum() / mel_mask.sum()

        # ===== Pitch loss =====
        pitch_pred = F.pad(pitch_pred, (0, pitch_tgt.size(2) - pitch_pred.size(2)), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask.unsqueeze(1)).sum() / dur_mask.sum()

        # ===== Energy loss =====
        if energy_pred is not None:
            energy_pred = F.pad(energy_pred, (0, energy_tgt.size(1) - energy_pred.size(1)), value=0.0)
            energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction='none')
            energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        else:
            energy_loss = torch.tensor(0.0, device=mel_out.device)

        # ===== Gate loss =====
        gate_out = F.pad(gate_out, (0, gate_tgt.size(1) - gate_out.size(1)), value=0.0)
        gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_tgt, reduction='none')
        gate_loss = (gate_loss * mel_mask).sum() / mel_mask.sum()

        # ===== Attention CTC loss =====
        attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        # ===== Final Loss =====
        total_loss = (
            self.mel_loss_scale * mel_loss +
            self.postnet_loss_scale * postnet_loss +
            self.gate_loss_scale * gate_loss +
            self.dur_loss_scale * dur_pred_loss +
            self.pitch_loss_scale * pitch_loss +
            self.energy_loss_scale * energy_loss +
            self.attn_loss_scale * attn_loss
        )

        # ===== Meta info =====
        meta = {
            'loss': total_loss.detach(),
            'mel_loss': mel_loss.detach(),
            'postnet_loss': postnet_loss.detach(),
            'gate_loss': gate_loss.detach(),
            'duration_loss': dur_pred_loss.detach(),
            'pitch_loss': pitch_loss.detach(),
            'energy_loss': energy_loss.detach(),
            'attn_loss': attn_loss.detach(),
            'dur_error': (torch.abs(dur_pred - attn_dur).sum() / dur_mask.sum()).detach()
        }

        return total_loss, meta
