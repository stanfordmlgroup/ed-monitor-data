import numpy as np
from scipy.signal import find_peaks, resample


def get_ptt(pleth, ii):
    """
    Retrieves the pulse transit time for two waveforms that are 125Hz
    Based on: https://github.com/rajpurkarlab/ed-monitor-decompensation/blob/2aa5d97cd4a56a4bb957e50aa26376ad5584f454/hrv_ptt/ptt.py
    """
    assert len(pleth) == len(ii)

    if np.any(pleth < 0) or np.any(ii < -10):
        # Simple quality check
        return np.nan

    pleth_search = pleth
    ii_search = ii
    ii_peaks, _ = find_peaks(ii_search, distance=37) # Assuming max 200 bpm @ 125 Hz => 125/(200/60) = 37.5
    pleth_peaks, _ = find_peaks(pleth_search, distance=37) # Assuming max 200 bpm @ 125 Hz => 125/(200/60) = 37.5
    ii_peaks_norm = ii_peaks * 8 # Assumes 125 Hz waveforms, so this normalizes to 1000ms
    pleth_peaks_norm = pleth_peaks * 8 # Assumes 125 Hz waveforms, so this normalizes to 1000ms

    ptts = []

    # get RR_ints
    ii_int = np.array([ii_peaks_norm[i + 1] - ii_peaks_norm[i] for i in range(0, len(ii_peaks_norm) - 1)])

    ii_peaks_used = []
    pleth_peaks_used = []
    # Now let's filter for just the peaks we know are good between ECG and PPG:
    while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0:

        # move pleth_peaks up one if we have an offset of peaks
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and ii_peaks_norm[0] > pleth_peaks_norm[0]:
            pleth_peaks_norm = pleth_peaks_norm[1:]
            break

        # if we have nothing left, break out
        if len(ii_peaks_norm) == 0 or len(pleth_peaks_norm) == 0:
            break

        # if we are too behind
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and pleth_peaks_norm[0] - ii_peaks_norm[0] > ii_int[
            0]:
            ii_peaks_norm = ii_peaks_norm[1:]
            if len(ii_int) > 1:
                ii_int = ii_int[1:]
            break

        # now we can get our PTTs
        ptt = pleth_peaks_norm[0] - ii_peaks_norm[0]
        if ptt < ii_int[0] and ptt > 0:
            ii_peaks_used.append(int(ii_peaks_norm[0] / 8))
            pleth_peaks_used.append(int(pleth_peaks_norm[0] / 8))
            ptts.append(ptt)

        # Update for next PTT
        ii_peaks_norm = ii_peaks_norm[1:]
        pleth_peaks_norm = pleth_peaks_norm[1:]
        if len(ii_int) > 1:
            ii_int = ii_int[1:]
        else:
            break

    if len(ptts) > 0:
        mean_ptt = np.mean(ptts)
        return float(mean_ptt)
    else:
        return np.nan
