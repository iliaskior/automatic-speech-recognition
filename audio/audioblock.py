from pathlib import Path
from typing import Union
import numpy as np
import librosa


def audio_load(filepath: Union[str, Path], sr: int) -> np.ndarray:
    """
    Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate

    Parameters
    ----------
    filepath : Union[str, Path]
        Local path of the audio file
    sr : int
        target sampling rate

    Returns
    -------
    np.ndarray
        audio time series
    """

    #Read audio file as floating point time series
    try:
        sig, sr = librosa.load(filepath, sr=sr)
    except Exception as e:
        print(f"An error occured while processing {filepath}")
        print(e)
        return None
    return sig

def get_duration(sig:np.ndarray, sr:int) -> float:
    """Returns the audio duration in seconds

    Parameters
    ----------
    sig : np.ndarray
        Target signal
    sr : int
        The sampling rate used for audio loading

    Returns
    -------
    float
        audio file duration
    """
    return sig.shape[0]/sr

def time_configure(sample_rate:int, time:int):
    """Convert time variable to corresponding number of samples according to sample rate

    Parameters
    ----------
    sample_rate : int
        The sampling rate of audio file
    time : int
        Time in milliseconds

    Returns
    -------
    int
        Number of samples
    """
    n_samples = int((time * sample_rate) / 1000)

    return n_samples

def split_to_signals(sig: np.ndarray, sr:int, size:int=2000, slide:int=2000) -> np.ndarray:
    """Cuts signal in segments of ``size`` length and ``slide`` overlap

    Parameters
    ----------
    signal : np.ndarray
        Input Signal
    sr : int
        The sampling rate of audio file
    size : int, optional
        Segment size in milliseconds, by default 4000
    slide : int, optional
        Overlap in milliseconds, by default 4000

    Returns
    -------
    np.ndarray
        Two-dimentional array of signal segments
    """
    
    if slide is None:
        slide = size

    assert size >= slide, "Size should be greater or equal than slide length"

    size_samples = time_configure(sr, size)
    slide_samples = time_configure(sr, slide)

    return librosa.util.frame(sig, frame_length=size_samples, hop_length=slide_samples, axis=0)


def extract_features(sig:np.ndarray, sr:int, num_mfcc:int=13,n_fft:int=128, hop_length:int=32) -> Union[np.ndarray,np.ndarray, np.ndarray]:
    """Extracts MFCCs, Delta-MFCCs and Delta-Delta-MFCCs

    Parameters
    ----------
    sig : np.ndarray
        Audio timeseries
    sr : int
        Sampling rate
    num_mfcc : int, optional
        Number of coefficients to extract, by default 13
    n_fft : int, optional
        Interval we consider to apply FFT. Measured in milliseconds, by default 128
    hop_length : int, optional
        Sliding window for FFT. Measured in milliseconds, by default 32

    Returns
    -------
    Union[np.ndarray,np.ndarray, np.ndarray]
        Extracted mfccs, delta-mfccs, delta2-mfccs
    """

    n_fft_samples = time_configure(sr, n_fft)
    hop_length_samples = time_configure(sr, hop_length)

    mfccs = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=num_mfcc, n_fft = n_fft_samples, hop_length = hop_length_samples)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfcss = librosa.feature.delta(mfccs, order=2)

    return mfccs, delta_mfccs, delta2_mfcss