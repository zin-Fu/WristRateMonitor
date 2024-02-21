import numpy as np
from scipy import signal

class Signal_processing():
    def __init__(self):
        self.a = 1
        
    def extract_color(self, ROIs):
        '''
        Extract the average green value from ROIs
        '''
        output_val = np.mean(ROIs)
        return output_val
    
    def normalization(self, data_buffer):
        '''
        Normalize the input data buffer
        '''
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        Remove the overall trend
        '''
        detrended_data = signal.detrend(data_buffer)
        return detrended_data
        
    def interpolation(self, data_buffer, times):
        '''
        Interpolate the data buffer to make the signal more periodic (avoid spectral leakage)
        '''
        L = len(data_buffer)
        even_times = np.linspace(times[0], times[-1], L)
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data
        
    def fft(self, data_buffer, fps):
        '''
        Perform Fast Fourier Transform on the data buffer
        '''
        L = len(data_buffer)
        freqs = float(fps) / L * np.arange(L / 2 + 1)
        freqs_in_minute = 60. * freqs
        raw_fft = np.fft.rfft(data_buffer*30)
        fft = np.abs(raw_fft)**2
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        interest_idx_sub = interest_idx[:-1].copy() # avoid the indexing error
        freqs_of_interest = freqs_in_minute[interest_idx_sub]
        fft_of_interest = fft[interest_idx_sub]
        return fft_of_interest, freqs_of_interest

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        '''
        Apply a Butterworth bandpass filter to the data buffer
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data_buffer)
        return filtered_data