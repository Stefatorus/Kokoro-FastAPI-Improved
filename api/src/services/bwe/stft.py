"""STFT/iSTFT utilities for bandwidth extension."""

import torch


def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
    """Convert audio to amplitude and phase spectrograms.

    Args:
        audio: Input audio tensor
        n_fft: FFT size
        hop_size: Hop length for STFT
        win_size: Window size
        center: Whether to center the STFT

    Returns:
        Tuple of (log_amplitude, phase, complex_representation)
    """
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    log_amp = torch.log(torch.abs(stft_spec)+1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp)*torch.cos(pha),
                       torch.exp(log_amp)*torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    """Convert amplitude and phase spectrograms back to audio.

    Args:
        log_amp: Log amplitude spectrogram
        pha: Phase spectrogram
        n_fft: FFT size
        hop_size: Hop length for iSTFT
        win_size: Window size
        center: Whether to center the iSTFT

    Returns:
        Audio tensor
    """
    amp = torch.exp(log_amp)
    com = torch.complex(amp*torch.cos(pha), amp*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio
