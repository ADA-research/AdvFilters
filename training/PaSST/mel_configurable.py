import torch
import torchaudio

EPSILON = 0.00001

window_functions = {
    "bartlett": torch.bartlett_window,
    "blackman": torch.blackman_window,
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
    "kaiser": torch.kaiser_window
}

class AugmentMelSTFT(torch.nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=True, f_min=0.0, f_max=None, window_fn="hann", power=2, normalized=False, center=True, 
                 pad_mode="reflect"): #, fmin_aug_range=1, fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        
        # Handle Params
        if f_max is None:
            #f_max = sr // 2 - fmax_aug_range // 2
            f_max = sr // 2
            print(f"WARNING: FMAX is None. Defaulting to {f_max} ")
        if htk:
            norm = None
            mel_scale = "htk"
        else: # use Slaney
            norm = "slaney"
            mel_scale = "slaney"
        
        if window_fn in window_functions:
            window_fn = window_functions[window_fn]
        else:
            print(f"WARNING: {window_fn} is not an available window function. Defaulting to hann.")
            window_fn = window_functions["hann"]
        
        # Create Mel Module
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                        n_fft=n_fft,
                                                        win_length=win_length,
                                                        hop_length=hopsize,
                                                        f_min=f_min,
                                                        f_max=f_max,
                                                        pad=0,
                                                        n_mels=n_mels,
                                                        window_fn=window_fn,
                                                        power=power,
                                                        normalized=normalized,
                                                        center=center,
                                                        pad_mode=pad_mode,
                                                        norm=norm,
                                                        mel_scale=mel_scale
                                                        )
        self.preemphasis_coef = torch.Tensor([[[-.97, 1]]]).half().cuda()
        
        # Masking Params
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)
        
    def forward(self, x): #TODO: Currently ommitting fmax randomization (see preprocess.py)
        x = torch.nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coef).squeeze(1)
        melspec = self.mel(x)
        melspec = (melspec + EPSILON).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec
    
    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )

