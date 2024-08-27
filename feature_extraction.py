import numpy as np
import scipy.signal as signal
from sklearn.decomposition import FastICA

def bandpower(data, sf, band, window_sec=4, relative=False):
    band = np.asarray(band)
    low, high = band
    nperseg = min(window_sec * sf, len(data))
    if nperseg < 2:
        return 0  # Return 0 if not enough data for bandpower calculation
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.sum(psd[idx_band]) * freq_res
    if relative:
        bp /= np.sum(psd)
    return bp

def extract_eeg_features(eeg_data, selected_channels, channel_names, fs=128, apply_ica=False):
    # Select specified channels by names
    channel_indices = [channel_names.index(ch) for ch in selected_channels]
    eeg_data = eeg_data[channel_indices, :]

    if apply_ica:
        ica = FastICA(n_components=eeg_data.shape[0], max_iter=1000, tol=0.01)
        eeg_data = ica.fit_transform(eeg_data.T).T

    bands = {
        "theta": [4, 8],
        "alpha": [8, 13],
        "low_beta": [13, 20],
        "high_beta": [20, 30],
        "delta": [0.5, 4]
    }
    features = []
    for channel in eeg_data:
        for band in bands:
            freq = bands[band]
            features.append(bandpower(channel, fs, freq))
        # FFT features
        fft_values = np.abs(np.fft.fft(channel))[:len(channel) // 2]
        features.extend(fft_values[:5])  # First 5 FFT components
    return np.array(features)

# Example usage
if __name__ == "__main__":
    # For preprocessed data (no ICA needed)
    preprocessed_eeg_sample = np.random.rand(2, 128)  # Example 2-channel EEG data with 128 samples
    channel_names = ['Fp1', 'Fp2']
    selected_channels = ['Fp1', 'Fp2']
    features_preprocessed = extract_eeg_features(preprocessed_eeg_sample, selected_channels, channel_names, fs=128, apply_ica=False)
    print("Extracted features (preprocessed data) shape:", features_preprocessed.shape)
    print(features_preprocessed)

    # For raw data (ICA applied)
    raw_eeg_sample = np.random.rand(2, 100)  # Example 2-channel raw EEG data with 100 samples
    features_raw = extract_eeg_features(raw_eeg_sample, selected_channels, channel_names, fs=100, apply_ica=True)
    print("Extracted features (raw data) shape:", features_raw.shape)
    print(features_raw)
