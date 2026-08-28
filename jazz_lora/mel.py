from __future__ import annotations

import torch

from .config import DataConfig, MelConfig


class AudioLDM2Mel(torch.nn.Module):
    """Waveform-to-log-mel transform matching the original AudioLDM2 dataset."""

    def __init__(self, data: DataConfig, mel: MelConfig) -> None:
        super().__init__()
        import librosa

        basis = librosa.filters.mel(
            sr=data.sample_rate,
            n_fft=mel.n_fft,
            n_mels=mel.n_mels,
            fmin=mel.f_min,
            fmax=mel.f_max,
        )
        self.register_buffer("mel_basis", torch.from_numpy(basis).float())
        self.register_buffer("window", torch.hann_window(mel.win_length))
        self.data = data
        self.mel = mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.float()
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        waveform = 0.5 * waveform / peak
        spectrum = torch.stft(
            waveform,
            n_fft=self.mel.n_fft,
            hop_length=self.mel.hop_length,
            win_length=self.mel.win_length,
            window=self.window.to(waveform.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        log_mel = torch.matmul(self.mel_basis.to(waveform.device), spectrum)
        log_mel = torch.log(log_mel.clamp_min(self.mel.log_clip))
        target_frames = int(
            self.data.clip_seconds * self.data.sample_rate / self.mel.hop_length
        )
        log_mel = log_mel.transpose(1, 2)
        if log_mel.shape[1] < target_frames:
            log_mel = torch.nn.functional.pad(
                log_mel, (0, 0, 0, target_frames - log_mel.shape[1])
            )
        log_mel = log_mel[:, :target_frames, :]
        if log_mel.shape[-1] % 2:
            log_mel = log_mel[..., :-1]
        return log_mel.unsqueeze(1)  # (batch, channel=1, time, mel)
