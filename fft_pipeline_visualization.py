import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def load_audio(file_path):
    """Load audio file and return signal and sample rate"""
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def pre_emphasis(signal, alpha=0.97):
    """Apply pre-emphasis filter"""
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal

def framing(signal, frame_length, hop_length):
    """Split signal into frames"""
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    return frames

def windowing(frames, window_type='hamming'):
    """Apply window function to frames"""
    if window_type == 'hamming':
        window = np.hamming(frames.shape[0])
    elif window_type == 'hanning':
        window = np.hanning(frames.shape[0])
    else:
        window = np.ones(frames.shape[0])
    
    windowed_frames = frames * window[:, np.newaxis]
    return windowed_frames

def compute_fft(windowed_frames, n_fft=2048):
    """Compute FFT for each frame"""
    fft_frames = np.abs(np.fft.rfft(windowed_frames, n=n_fft, axis=0))
    return fft_frames

def aggregate_spectrum(fft_frames):
    """Calculate mean spectrum across all frames"""
    mean_spectrum = np.mean(fft_frames, axis=1)
    return mean_spectrum

def log_scaling(spectrum):
    """Apply log scaling to spectrum"""
    log_spectrum = np.log(spectrum + 1e-10)  # Add small value to avoid log(0)
    return log_spectrum

def plot_pipeline_steps(file_path):
    """Create plots for each step of the FFT pipeline"""
    
    # Load audio
    signal, sr = load_audio(file_path)
    
    # Parameters
    frame_length = 1024
    hop_length = 512
    n_fft = 2048
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    
    # Step 1: Original Signal
    plt.subplot(4, 2, 1)
    time_axis = np.linspace(0, len(signal)/sr, len(signal))
    plt.plot(time_axis, signal)
    plt.title('Step 1: Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Step 2: Pre-emphasis
    emphasized_signal = pre_emphasis(signal)
    plt.subplot(4, 2, 2)
    plt.plot(time_axis, emphasized_signal)
    plt.title('Step 2: Pre-emphasis Filter')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Step 3: Framing
    frames = framing(emphasized_signal, frame_length, hop_length)
    plt.subplot(4, 2, 3)
    # Show first few frames
    for i in range(min(5, frames.shape[1])):
        frame_start = i * hop_length
        frame_end = frame_start + frame_length
        time_frame = np.linspace(frame_start/sr, frame_end/sr, frame_length)
        plt.plot(time_frame, frames[:, i], alpha=0.7, label=f'Frame {i+1}')
    plt.title('Step 3: Framing (First 5 frames)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Step 4: Windowing
    windowed_frames = windowing(frames)
    plt.subplot(4, 2, 4)
    # Show windowed version of first frame
    frame_start = 0
    frame_end = frame_start + frame_length
    time_frame = np.linspace(frame_start/sr, frame_end/sr, frame_length)
    plt.plot(time_frame, frames[:, 0], alpha=0.5, label='Original frame')
    plt.plot(time_frame, windowed_frames[:, 0], label='Windowed frame')
    plt.title('Step 4: Windowing (First frame)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Step 5: FFT
    fft_frames = compute_fft(windowed_frames, n_fft)
    plt.subplot(4, 2, 5)
    # Show magnitude spectrum of first few frames
    freq_axis = np.linspace(0, sr/2, fft_frames.shape[0])
    for i in range(min(3, fft_frames.shape[1])):
        plt.plot(freq_axis, fft_frames[:, i], alpha=0.7, label=f'Frame {i+1}')
    plt.title('Step 5: FFT Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Step 6: Aggregation
    mean_spectrum = aggregate_spectrum(fft_frames)
    plt.subplot(4, 2, 6)
    plt.plot(freq_axis, mean_spectrum)
    plt.title('Step 6: Aggregated Spectrum (Mean)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Step 7: Log Scaling
    log_spectrum = log_scaling(mean_spectrum)
    plt.subplot(4, 2, 7)
    plt.plot(freq_axis, log_spectrum)
    plt.title('Step 7: Log-scaled Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Final Feature Vector
    plt.subplot(4, 2, 8)
    plt.plot(log_spectrum)
    plt.title('Final FFT Feature Vector')
    plt.xlabel('Feature Index')
    plt.ylabel('Log Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fft_pipeline_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return signal, emphasized_signal, frames, windowed_frames, fft_frames, mean_spectrum, log_spectrum

# Main execution
if __name__ == "__main__":
    file_path = r"D:\Queenless\20k_audio_splitted_dataset\train\NonQueen\queenless_34.wav"
    
    if os.path.exists(file_path):
        print("Processing audio file...")
        results = plot_pipeline_steps(file_path)
        print("Visualization complete! Check 'fft_pipeline_visualization.png'")
    else:
        print(f"File not found: {file_path}")





