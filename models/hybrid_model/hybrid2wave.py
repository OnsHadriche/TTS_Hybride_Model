from typing import List, Union, Optional, Literal
import torch
import torch.nn as nn

from utils import get_basic_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser
# Import your hybrid TTS model
from hybride_model import HybrideTTS
# Adjust the import path above as needed

_VOWELIZER_TYPE = Literal['shakkala', 'shakkelha']

class HybrideTTS2Wave(nn.Module):
    def __init__(self,
                 model_sd_path: str,
                 vocoder_sd: Optional[str] = None,
                 vocoder_config: Optional[str] = None,
                 # Additional options if needed:
                 vowelizer: Optional[_VOWELIZER_TYPE] = None,
                 ):
        super().__init__()
        # Load basic config
        self.config = get_basic_config()

        # Load your hybrid TTS model's state dict
        state_dicts = torch.load(model_sd_path, map_location='cpu')
        # Create your model instance with the same hyperparameters (adjust as needed)
        self.model = HybrideTTS(
            n_mel_channels=self.config.n_mel_channels,
            n_symbols=self.config.n_symbols,
            padding_idx=self.config.padding_idx,
            symbols_embedding_dim=self.config.symbols_embedding_dim,
            in_fft_n_layers=self.config.in_fft_n_layers,
            in_fft_n_heads=self.config.in_fft_n_heads,
            in_fft_d_head=self.config.in_fft_d_head,
            in_fft_conv1d_kernel_size=self.config.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=self.config.in_fft_conv1d_filter_size,
            in_fft_output_size=self.config.in_fft_output_size,
            p_in_fft_dropout=self.config.p_in_fft_dropout,
            p_in_fft_dropatt=self.config.p_in_fft_dropatt,
            p_in_fft_dropemb=self.config.p_in_fft_dropemb,
            out_fft_n_layers=self.config.out_fft_n_layers,
            out_fft_n_heads=self.config.out_fft_n_heads,
            out_fft_d_head=self.config.out_fft_d_head,
            out_fft_conv1d_kernel_size=self.config.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=self.config.out_fft_conv1d_filter_size,
            out_fft_output_size=self.config.out_fft_output_size,
            p_out_fft_dropout=self.config.p_out_fft_dropout,
            p_out_fft_dropatt=self.config.p_out_fft_dropatt,
            p_out_fft_dropemb=self.config.p_out_fft_dropemb,
            dur_predictor_kernel_size=self.config.dur_predictor_kernel_size,
            dur_predictor_filter_size=self.config.dur_predictor_filter_size,
            p_dur_predictor_dropout=self.config.p_dur_predictor_dropout,
            dur_predictor_n_layers=self.config.dur_predictor_n_layers,
            pitch_predictor_kernel_size=self.config.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=self.config.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=self.config.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=self.config.pitch_predictor_n_layers,
            pitch_embedding_kernel_size=self.config.pitch_embedding_kernel_size,
            energy_conditioning=self.config.energy_conditioning,
            energy_predictor_kernel_size=self.config.energy_predictor_kernel_size,
            energy_predictor_filter_size=self.config.energy_predictor_filter_size,
            p_energy_predictor_dropout=self.config.p_energy_predictor_dropout,
            energy_predictor_n_layers=self.config.energy_predictor_n_layers,
            energy_embedding_kernel_size=self.config.energy_embedding_kernel_size,
            n_speakers=self.config.n_speakers,
            speaker_emb_weight=self.config.speaker_emb_weight,
            pitch_conditioning_formants=self.config.pitch_conditioning_formants
        )
        # Load the model weights
        self.model.load_state_dict(state_dicts['model'], strict=False)
        self.model.eval()

        # Load vocoder if paths are not provided, get them from config
        if vocoder_sd is None or vocoder_config is None:
            vocoder_sd = self.config.vocoder_state_path
            vocoder_config = self.config.vocoder_config_path

        vocoder = load_hifigan(vocoder_sd, vocoder_config)
        self.vocoder = vocoder
        self.denoiser = Denoiser(vocoder)

        # Optionally, load vowelizer(s) if your hybrid model supports it
        self.vowelizer = vowelizer
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def tts_single(self,
                   text_input: str,
                   speed: float = 1.0,
                   speaker_id: int = 0,
                   denoise: float = 0.0,
                   pitch_mul: float = 1.0,
                   pitch_add: float = 0.0,
                   ) -> torch.Tensor:
        """
        Converts a single text utterance into a waveform.
        """
        # Here we assume your HybrideTTS model has a method `ttmel_single`
        # which returns mel spectrogram output.
        mel_spec = self.model.ttmel_single(
            text_input,
            speed=speed,
            speaker_id=speaker_id,
            vowelizer=self.vowelizer,
            pitch_mul=pitch_mul,
            pitch_add=pitch_add
        )
        # Pass mel spectrogram through vocoder to get waveform.
        wave = self.vocoder(mel_spec)
        if denoise > 0:
            wave = self.denoiser(wave, denoise)
        return wave[0].cpu()

    @torch.inference_mode()
    def tts_batch(self,
                  texts: List[str],
                  speed: float = 1.0,
                  speaker_id: int = 0,
                  denoise: float = 0.0,
                  pitch_mul: float = 1.0,
                  pitch_add: float = 0.0,
                  batch_size: int = 1) -> List[torch.Tensor]:
        """
        Converts a batch of text utterances into a list of waveforms.
        """
        mel_list = self.model.ttmel(texts,
                                    speed=speed,
                                    speaker_id=speaker_id,
                                    batch_size=batch_size,
                                    vowelizer=self.vowelizer,
                                    pitch_mul=pitch_mul,
                                    pitch_add=pitch_add)
        wav_list = []
        for mel in mel_list:
            wav = self.vocoder(mel)
            if denoise > 0:
                wav = self.denoiser(wav, denoise)
            wav_list.append(wav[0].cpu())
        return wav_list

    def tts(self,
            text_input: Union[str, List[str]],
            speed: float = 1.0,
            speaker_id: int = 0,
            batch_size: int = 1,
            denoise: float = 0.005,
            pitch_mul: float = 1.0,
            pitch_add: float = 0.0,
            ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Main method to synthesize audio from text using the hybrid model.
        Accepts a single string or a list of strings.
        Returns a waveform (or list of waveforms).
        """
        if isinstance(text_input, str):
            return self.tts_single(text_input,
                                   speed=speed,
                                   speaker_id=speaker_id,
                                   denoise=denoise,
                                   pitch_mul=pitch_mul,
                                   pitch_add=pitch_add)
        else:
            return self.tts_batch(text_input,
                                  speed=speed,
                                  speaker_id=speaker_id,
                                  denoise=denoise,
                                  pitch_mul=pitch_mul,
                                  pitch_add=pitch_add,
                                  batch_size=batch_size)
