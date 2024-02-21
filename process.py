import cv2
import numpy as np
import time
from scipy import signal
from signal_processing import Signal_processing

def wrist_detect(img):

    height, width, _ = img.shape
    start_x = width // 2 - 100
    start_y = height // 2 - 100
    end_x = width // 2 + 100
    end_y = height // 2 + 100

    # Recognize the rectangular area of the ROI
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
    cv2.putText(img, 'PUT YOUR WRIST HERE', (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    # Crop the image to the rectangular area
    img_roi = img[start_y:end_y, start_x:end_x]

    # Color space conversion
    img_HSV = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    img_YCrCb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Merge
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    # Find the largest contour
    contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ROI = img_roi
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(global_mask)
        cv2.drawContours(contour_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
        ROI = cv2.bitwise_and(img_roi, img_roi, mask=contour_mask)

    return ROI

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)  # input frame
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)  # Region of Interest frame
        self.frame_out = np.zeros((10, 10, 3), np.uint8)  # output frame
        self.samples = []  # list to store samples
        self.buffer_size = 100  # size of the buffer
        self.times = []  # list to store time
        self.data_buffer = []  # buffer to store data
        self.fps = 0  # frames per second
        self.fft = []  # Fast Fourier Transform
        self.freqs = []  # frequencies
        self.t0 = time.time()  # start time
        self.bpm = 0  # beats per minute
        self.bpms = []  # list to store beats per minute
        self.peaks = []  # list to store peaks
        self.sp = Signal_processing()  # instance of Signal_processing class

    def extractColor(self, frame):
        g = np.mean(frame[:,:,1])  # extract green color
        return g
        
    def run(self):
        frame = self.frame_in
        ROIs = wrist_detect(frame)  # detect wrist in the frame
        green_val = self.sp.extract_color(ROIs)  # extract green color from ROIs
        self.frame_out = frame  # assign current input frame to output frame for display and saving
        L = len(self.data_buffer)  # length of data_buffer
        g = green_val  # extract green color from ROIs (short form)
        
        # remove sudden changes, if avg value changes more than 10, use the average value of data_buffer
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): 
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)  # append the time difference to times list
        self.data_buffer.append(g)  # add the average value extracted from the green channel to the data buffer list

        # can only process in a fixed size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 10 frames
        if L == self.buffer_size:
            self.fps = float(L) / (self.times[-1] - self.times[0])  # calculate HR using the real fps of the computer processor, not the fps provided by the camera
            even_times = np.linspace(self.times[0], self.times[-1], L)
            processed = signal.detrend(processed)  # eliminate the trend of the signal to avoid interference from light changes
            interpolated = np.interp(even_times, self.times, processed)  # interpolation by 1
            interpolated = np.hamming(L) * interpolated  # make the signal more periodic (avoid spectral leakage)
            norm = interpolated/np.linalg.norm(interpolated)  # normalization
            raw = np.fft.rfft(norm*30)  # do real fft with the normalization multiplied by 10
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            self.fft = np.abs(raw)**2  # get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within 
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            idx2 = np.argmax(pruned)  # max in the range can be HR
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)  # call the butter_bandpass_filter method in the Process class
        self.samples = processed  # multiply the signal with 5 for easier to see in the plot

        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
