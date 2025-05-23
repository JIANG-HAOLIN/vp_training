import torch
import torchaudio


class MelSpec(torch.nn.Module):
    """Compute mel spectrogram for audio signal"""
    def __init__(self,
                 windows_size: float = 0.025,
                 length: int = 40000,
                 sr: int = 16000,  # originally 44.1k, resampled to 16k
                 n_mels: int = 64,
                 norm_audio: bool = False,
                 hop: float = 0.01):
        """
        length - number of sample in an input sequence
        sr - sampling rate for mel spectrogram
        n_mel - Number of mel filterbanks
        norm_audio - whether do we normalize the audio signal
        """
        super().__init__()
        self.norm_audio = norm_audio
        self.n_mel = n_mels
        hop_length = int(sr * hop)
        n_fft = int(sr * windows_size)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.out_size = (n_mels, int(length/hop_length)+1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
                waveform - the input sound signal in waveform
        Return:
                log_spec - the logarithm form of mel spectrogram of sound signal
        """
        eps = 1e-8
        spec = self.mel(waveform.float())
        log_spec = torch.log(spec + eps)  # logarithm, eps to avoid zero denominator
        assert log_spec.size(-2) == self.n_mel
        if self.norm_audio:
            log_spec /= log_spec.sum(dim=-2, keepdim=True)
        return log_spec
