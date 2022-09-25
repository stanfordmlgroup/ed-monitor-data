# 
# Adapted from: https://github.com/meta00/vital_sqi/tree/main/vital_sqi
#
"""Signal quality indexes based on dynamic template matching"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew, entropy
from edm.sqi.band_filter import BandpassFilter
from edm.sqi.rpeak_detection import PeakDetector

"""
Most of the sqi scores are obtained from the following paper Elgendi,
Mohamed, Optimal signal quality index for photoplethysmogram
signals, Bioengineering.
"""


def tapering(signal_data,window=None,shift_min_to_zero=True):
    """
    expose
    Pin the leftmost and rightmost signal to the zero baseline
    and amplify the remainder according to the window shape
    :param signal_data: list,
    :param window:sequence, array of floats indicates the windows types
    as described in scipy.windows
    :return: the tapered signal
    """
    if shift_min_to_zero:
        signal_data = signal_data-np.min(signal_data)
    if window == None:
        window = signal.windows.tukey(len(signal_data),0.9)
    signal_data_tapered = np.array(window) * (signal_data)
    return np.array(signal_data_tapered)



def perfusion_sqi(x, y):
    """The perfusion index is the ratio of the pulsatile blood flow to the
    nonpulsatile or static blood in peripheral tissue.
    In other words, it is the difference of the amount of light absorbed
    through the pulse of when light is transmitted through the finger,
    which can be defined as follows:
    PSQI=[(ymax−ymin)/x¯|]×100 where PSQI is the perfusion index, x¯ is the
    statistical mean of the x signal (raw PPG signal), and y is the filtered
    PPG signal

    Parameters
    ----------
    x :
        float, mean of the raw signal
    y :
        list, array of filtered signal

    Returns
    -------

    """
    return (np.max(y)-np.min(y))/np.abs(np.mean(x))*100


def kurtosis_sqi(x, axis=0, fisher=True, bias=True,
                 nan_policy='propagate'):
    """Expose
    Kurtosis is a measure of whether the data are heavy-tailed or
    light-tailed relative to a normal distribution. That is, data sets with
    high kurtosis tend to have heavy tails, or outliers.
    Data sets with low kurtosis tend to have light tails, or lack of outliers.
    A uniform distribution would be the extreme case.

    Kurtosis is a statistical measure used to describe the distribution of
    observed data around the mean. It represents a heavy tail and peakedness
    or a light tail and flatness of a distribution relative to the normal
    distribution, which is defined as:

    Parameters
    ----------
    x :
        list, the array of signal
    axis :
         (Default value = 0)
    fisher :
         (Default value = True)
    bias :
         (Default value = True)
    nan_policy :
         (Default value = 'propagate')

    Returns
    -------

    """

    return kurtosis(x, axis, fisher, bias, nan_policy)


def skewness_sqi(x, axis=0, bias=True, nan_policy='propagate'):
    """Expose
    Skewness is a measure of symmetry, or more precisely, the lack of
    symmetry. A distribution, or data set, is symmetric if it looks the same
    to the left and right of the center point.

    Skewness is a measure of the symmetry (or the lack of it) of a
    probability distribution, which is defined as:
    SSQI=1/N∑i=1N[xi−μˆx/σ]3
    where μˆx and σ are the empirical estimate of the mean and standard
    deviation of xi,respectively; and N is the number of samples in the PPG
    signal.

    Parameters
    ----------
    x :
        list, the array of signal
    axis :
         (Default value = 0)
    bias :
         (Default value = True)
    nan_policy :
         (Default value = 'propagate')

    Returns
    -------

    """
    return skew(x, axis, bias, nan_policy)


def entropy_sqi(x, qk=None, base=None, axis=0):
    """Expose
    Calculate the entropy information from the template distribution. Using
    scipy package function.

    Parameters
    ----------
    x :
        list the input signal
    qk :
        list, array against which the relative entropy
        is computed (Default value = None)
    base :
        float, (Default value = None)
    axis :
        return: (Default value = 0)

    Returns
    -------

    """
    x_ = x - min(x)
    return entropy(x_, qk, base, axis)


def signal_to_noise_sqi(a, axis=0, ddof=0):
    """Expose
    A measure used in science and engineering that compares the level of a
    desired signal to the level of background noise.

    Parameters
    ----------
    a :
        param axis:
    ddof :
        return: (Default value = 0)
    axis :
         (Default value = 0)

    Returns
    -------

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def zero_crossings_rate_sqi(y, threshold=1e-10, ref_magnitude=None,
                            pad=True, zero_pos=True, axis=-1):
    """Reuse the function from librosa package.
    This is the rate of sign-changes in the processed signal, that is,
    the rate at which the signal changes from positive to negative or back.

    Parameters
    ----------
    y :
        list, array of signal
    threshold :
        float > 0, default=1e-10 if specified, values where
        -threshold <= y <= threshold are clipped to 0.
    ref_magnitude :
        float >0 If numeric, the threshold is scaled
        relative to ref_magnitude.
        If callable, the threshold is scaled relative
        to ref_magnitude(np.abs(y)). (Default value = None)
    pad :
        boolean, if True, then y[0] is considered a valid
        zero-crossing. (Default value = True)
    zero_pos :
        the crossing marker. (Default value = True)
    axis :
        axis along which to compute zero-crossings. (Default value = -1)

    Returns
    -------
    type
        float, indicator array of zero-crossings in `y` along the
        selected axis.

    """
    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    crossings = np.pad(
        (y_sign[tuple(slice_post)] != y_sign[tuple(slice_pre)]),
        padding,
        mode="constant",
        constant_values=pad,
    )

    return np.mean(crossings, axis=0, keepdims=True)[0]


def mean_crossing_rate_sqi(y, threshold=1e-10, ref_magnitude=None,
                           pad=True, zero_pos=True, axis=-1):
    """Expose
    Same as zero crossing rate but this function interests in the rate of
    crossing signal mean

    Parameters
    ----------
    y :
        param threshold:
    ref_magnitude :
        param pad: (Default value = None)
    zero_pos :
        param axis: (Default value = True)
    threshold :
         (Default value = 1e-10)
    pad :
         (Default value = True)
    axis :
         (Default value = -1)

    Returns
    -------

    """
    return zero_crossings_rate_sqi(y-np.mean(y), threshold, ref_magnitude,
                                   pad, zero_pos, axis)


def msq_sqi(y, peaks_1, peak_detect2=6):
    """
    MSQ SQI as defined in Elgendi et al
    "Optimal Signal Quality Index for Photoplethysmogram Signals"
    with modification of the second algorithm used.
    Instead of Bing's, a SciPy built-in implementation is used.
    The SQI tracks the agreement between two peak detectors
    to evaluate quality of the signal.

    Parameters
    ----------
    x : sequence
        A signal with peaks.

    peaks_1 : array of int  
        Already computed peaks arry from the primary peak_detector

    peak_detect2 : int
        Type of the second peak detection algorithm, default = Scipy

    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal

    """
    detector = PeakDetector(wave_type='ppg')
    peaks_2,_ = detector.ppg_detector(y, detector_type=peak_detect2, preprocess=False)
    if len(peaks_1)==0 or len(peaks_2)==0:
        return 0.0
    peak1_dom = len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)
    peak2_dom = len(np.intersect1d(peaks_2,peaks_1))/len(peaks_2)
    return min(peak1_dom, peak2_dom)

def per_beat_sqi(sqi_func, troughs, signal, taper, **kwargs):
    """
    Perform a per-beat application of the selected SQI function on the signal segment

    Parameters
    ----------
    sqi_func : function 
        An SQI function to be performed.

    troughs : array of int  
        Idices of troughs in the signal provided by peak detector to be able to extract individual beats

    signal : 
        Signal array containing one segment of the waveform

    taper : bool
        Is each beat need to be tapered or not before executing the SQI function

    **kwargs : dict
        Additional positional arguments that needs to be fed into the SQI function

    Returns
    -------
    calculated_SQI : array
        An array with SQI values for each beat of the signal

    """
    #Remove first and last trough as they might be on the edge
    troughs = troughs[1:-1]
    if len(troughs) > 2:
        sqi_vals = []
        for idx, beat_start in enumerate(troughs[:-1]):
            single_beat = signal[beat_start:troughs[idx+1]]
            if taper:
                single_beat = tapering(single_beat)
            if len(kwargs) != 0:
                args = tuple(kwargs.values()) 
                sqi_vals.append(sqi_func(single_beat, *args))
            else:
                sqi_vals.append(sqi_func(single_beat))
        return sqi_vals

    else:
        return -np.inf
        raise Exception("Not enough peaks in the signal to generate per beat SQI")


def contains_stationary_segments(data, window_length=25, window_skip=25):
    """
    Returns true if it finds completely stationary segments
    https://www.frontiersin.org/articles/10.3389/fdgth.2022.847555/full
    """
    window_matrix = np.lib.stride_tricks.sliding_window_view(data, window_length)[::window_skip]
    for window in window_matrix:
        if round(np.amax(window)) == round(np.amin(window)):
            return True
    return False


def segment_PPG_SQI_extraction(signal_segment, sampling_rate=125, primary_peakdet=7, secondary_peakdet=6, hp_cutoff_order=(1, 1), lp_cutoff_order=(20, 4), template_type=1):
    """
    Extract all package available SQIs from a single segment of PPG waveform. Return a dataframe with all SQIs and cut points for each segment.
    Parameters
    ----------
    signal_segment : array-like
    A segment of raw signal. The length is user defined in compute_SQI() function
    sampling_rate : int
    Sampling rate of the signal
    primary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The primary one is used to segment the waveform
    secondary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The secondary peakdetector is used to compute MSQ SQI
    hp_cutoff_order : touple (int, int)
    A high pass filter parameters, cutoff frequency and order
    Lp_cutoff_order : touple (int, int)
    A low pass filter parameters, cutoff frequency and order
    
    template_type : int
    Selects which template from the dtw SQI should be used       
    Returns
    -------
    Pandas series object with all SQIs for the given segment
    """
    raw_segment = signal_segment
    #Prepare final dictonary that will be converted to dataFrame at the end
    SQI_dict = {}
    #Prepare filter and filter signal
    filt = BandpassFilter(band_type='butter', fs=sampling_rate)
    filtered_segment = filt.signal_highpass_filter(raw_segment, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
    filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    #Prepare primary peak detector and perform peak detection
    detector = PeakDetector()
    peak_list, trough_list = detector.ppg_detector(filtered_segment, primary_peakdet)
    #Helpful lists for iteration
    variations_stats = ['', '_mean', '_median', '_std']
    variations_acf = ['_peak1', '_peak2', '_peak3', '_value1', '_value2', '_value3']
    stats_functions = [('skewness', skewness_sqi), ('kurtosis', kurtosis_sqi), ('entropy', entropy_sqi)]
    #Raw signal SQI computation
    SQI_dict['snr'] = np.mean(signal_to_noise_sqi(raw_segment))
    SQI_dict['perfusion'] = perfusion_sqi(y=filtered_segment, x=raw_segment)
    SQI_dict['mean_cross'] = mean_crossing_rate_sqi(raw_segment)
    #Filtered signal SQI computation
    SQI_dict['zero_cross'] = zero_crossings_rate_sqi(filtered_segment)
    SQI_dict['msq'] = msq_sqi(y=filtered_segment, peaks_1=peak_list, peak_detect2=secondary_peakdet)

    for funcion in stats_functions:
        SQI_dict[funcion[0]+variations_stats[0]] = funcion[1](filtered_segment)
        statSQI_list = per_beat_sqi(sqi_func=funcion[1], troughs=trough_list, signal=filtered_segment, taper=True)
        SQI_dict[funcion[0]+variations_stats[1]] = np.mean(statSQI_list)
        SQI_dict[funcion[0]+variations_stats[2]] = np.median(statSQI_list)
        SQI_dict[funcion[0]+variations_stats[3]] = np.std(statSQI_list)

    return SQI_dict