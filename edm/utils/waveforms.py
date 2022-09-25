import biosppy
import numpy as np
from biosppy.signals.tools import filter_signal
from scipy import signal
from scipy.signal import decimate, resample
from edm.sqi.band_filter import BandpassFilter
from edm.sqi.standard_sqi import contains_stationary_segments, segment_PPG_SQI_extraction

WAVEFORMS_OF_INTERST = {
    "II": {
        "orig_frequency": 500,
        "bandpass_type": 'filter',
        "bandpass_freq": [3, 45]
    },
    "Pleth": {
        "orig_frequency": 125,
        "bandpass_type": 'butter',
        "bandpass_freq": None
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


def butter_bandpass(raw_segment, sampling_rate=125):
    hp_cutoff_order = (1, 1)
    lp_cutoff_order = (20, 4)
    filt = BandpassFilter(band_type='butter', fs=sampling_rate)
    filtered_segment = filt.signal_highpass_filter(raw_segment, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
    filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    return filtered_segment


def get_waveform(waveform, start, window_sz, orig_fs, should_normalize=False, bandpass_type=None,
                 bandwidth=[3, 45], target_fs=None, waveform_type="II", skewness_max=0.4, msq_min=0.25):
    waveform = waveform[(start):(start + window_sz)]
    if bandpass_type == "cheby2":
        # Recommended by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6316358/
        b, a = signal.cheby2(4, 40, bandwidth, 'bandpass', fs=orig_fs, output='ba')
        waveform = signal.lfilter(b, a, waveform)
    elif bandpass_type == "filter":
        waveform = apply_filter(waveform, filter_bandwidth=bandwidth, fs=orig_fs)
    elif bandpass_type == "butter":
        waveform = butter_bandpass(waveform, sampling_rate=orig_fs)

    if should_normalize:
        bottom, top = np.percentile(waveform, [1, 99])
        waveform = np.clip(waveform, bottom, top)
        waveform = normalize(waveform)

    if target_fs is not None:
        if orig_fs > target_fs or orig_fs < target_fs:
            waveform = resample(waveform, int(waveform.shape[-1] * (target_fs / orig_fs)))
        waveform = np.squeeze(waveform)
    else:
        target_fs = orig_fs

    failed_selection = False
    penalty = 0
    if waveform_type == "II":
        #
        # ECG waveform selection strategy
        #
        
        # Check for extreme outliers in the ECG
        #
        if max(waveform) > 4 or min(waveform) < -4:
            failed_selection = True

        # Check for enough heart beats
        # Note: 25 BPM from https://www.frontiersin.org/articles/10.3389/fdgth.2022.847555/full
        #
        try:
            hr_mean, hr_std = extract_ecg_infos(waveform, target_fs)
            if hr_mean < 25 or hr_mean >= 300:
                # Not enough or too many heart beats detected, indicating that the waveform was not a good ECG
                failed_selection = True
        except Exception:
            # The biosppy library will throw an error when it cannot find any heart beats
            failed_selection = True
        
        # Check for completely flat-lined waveform
        #
        if abs(sum(np.diff(waveform))) == 0:
            failed_selection = True
    elif waveform_type == "Pleth":
        #
        # PPG waveform selection strategy
        #
        
        # Check that skew and MSQ are within acceptable bounds (hyperparameter)
        # 
        ppg_assess = segment_PPG_SQI_extraction(waveform, target_fs)
        if abs(ppg_assess['skewness_mean']) >= skewness_max or ppg_assess['msq'] < msq_min:
            failed_selection = True
        
        # Check for stationary segments (e.g. flat-line)
        # Note that this is an expensive call, so we prefer not to run it if possible
        #
        if not failed_selection and contains_stationary_segments(waveform):
            failed_selection = True

    else:
        # Not yet implemented
        pass

    if not failed_selection:
        return waveform, 1
    else:
        return waveform, 0
