import biosppy
import numpy as np
from biosppy.signals.tools import filter_signal
from scipy import signal
from scipy.signal import decimate, resample

WAVEFORMS_OF_INTERST = {
    "II": {
        "orig_frequency": 500,
        "bandpass_type": 'filter',
        "bandpass_freq": [3, 45]
    },
    "Pleth": {
        "orig_frequency": 125,
        "bandpass_type": None,
        "bandpass_freq": [3, 45]
    },
    "Resp": {
        "orig_frequency": 62.5,
        "bandpass_type": 'cheby2',
        "bandpass_freq": [0.5, 10]
    }
}


def normalize(seq, smooth=1e-8):
    """
    Normalize each sequence between -1 and 1
    """
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq) + smooth) - 1


def apply_filter(signal, filter_bandwidth=[3, 45], fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    try:
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth,
                                     sampling_rate=fs)
    except:
        pass

    return signal


def extract_ecg_infos(a, fs):
    """
    ts (array) – Signal time axis reference (seconds).
    filtered (array) – Filtered ECG signal.
    rpeaks (array) – R-peak location indices.
    templates_ts (array) – Templates time axis reference (seconds).
    templates (array) – Extracted heartbeat templates.
    heart_rate_ts (array) – Heart rate time axis reference (seconds).
    heart_rate (array) – Instantaneous heart rate (bpm).
    """
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(a, sampling_rate=fs, show=False)

    if len(hr) == 0:
        return float("nan"), float("nan")
    else:
        return np.mean(hr), np.std(hr)


def get_waveform(waveform, start, window_sz, orig_fs, should_normalize=False, bandpass_type=None,
                 bandwidth=[3, 45], target_fs=None, ecg_quality_check=False):
    waveform = waveform[(start):(start + window_sz)]
    if bandpass_type == "cheby2":
        # Recommended by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6316358/
        b, a = signal.cheby2(4, 40, bandwidth, 'bandpass', fs=orig_fs, output='ba')
        waveform = signal.lfilter(b, a, waveform)
    elif bandpass_type == "filter":
        waveform = apply_filter(waveform, filter_bandwidth=bandwidth, fs=orig_fs)
    if should_normalize:
        bottom, top = np.percentile(waveform, [1, 99])
        waveform = np.clip(waveform, bottom, top)
        waveform = normalize(waveform)

    if target_fs is not None:
        # Standardize sampling rate
        if orig_fs > target_fs:
            waveform = decimate(waveform, int(orig_fs / target_fs))
        elif orig_fs < target_fs:
            waveform = resample(waveform, int(waveform.shape[-1] * (target_fs / orig_fs)))
        waveform = np.squeeze(waveform)
    else:
        target_fs = orig_fs

    failed_selection = False
    if ecg_quality_check:
        try:
            hr_mean, hr_std = extract_ecg_infos(waveform, target_fs)
            if hr_mean < 10:
                # Not enough heart beats detected, indicating that the waveform was not a good ECG
                failed_selection = True
        except Exception:
            # The biosppy library will throw an error when it cannot find any heart beats
            failed_selection = True

    # Did we sample an empty waveform?
    if abs(sum(np.diff(waveform))) > 0 and not failed_selection:
        # Non-empty waveform
        return waveform, 1
    else:
        return waveform, 0
