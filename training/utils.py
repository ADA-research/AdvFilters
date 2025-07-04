import av
import io
import numpy as np
from sklearn.metrics import average_precision_score
import torch

def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    try:
        container = av.open(io.BytesIO(mp3_arr.tobytes()))
    except ValueError as e:
        print("Error in decode_mp3")
        print(e)
        return np.zeros(320000)
    stream = next(s for s in container.streams if s.type == 'audio')
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.pad(x, (0, audio_length - len(x)))
        #return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[:audio_length] # Truncate 
    
def mixup(dataset, x, y, beta=2, rate=0.5):
    """Mixup adapted from Koutini et. al. (PaSST)"""
    if torch.rand(1) < rate:
        idx2 = torch.randint(len(dataset), (1,)).item()
        x2, y2 = dataset[idx2]
        l = np.random.beta(beta, beta)
        l = max(l, 1. - l)
        x1 = x-x.mean()
        x2 = x2-x2.mean()
        x = (x1 * l + x2 * (1. - l))
        x = x - x.mean()
        y = y * l + y2 * (1. - l)
    return x, y
    
def roll(x, shift_range=50):
    shift = int(np.random.random_integers(-shift_range, shift_range))
    return x.roll(shift, 0)
    
def masked_mean_average_precision(targets, preds, masks):
    try:
        targets = targets.round()
        ap_scores = []
        for i in range(preds.shape[1]):
            tar = targets[:, i]
            pre = preds[:, i]
            mas = masks[:, i]
            ap_score = average_precision_score(tar, pre, sample_weight=mas) # Koutini et. al. 2022
            ap_scores.append(ap_score)
        # Return the mean of AP across all valid classes (this is mAP)
        return np.mean(ap_scores)
    except:
        print("Error calculating masked mAP. Returning 0.")
        return 0