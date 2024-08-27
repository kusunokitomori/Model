import numpy as np
from load_data import load_deap_data
from feature_extraction import extract_eeg_features


def prepare_dataset(file_paths, selected_channels, fs=128):
    X_list = []
    y_list = []

    for file_path in file_paths:
        data, labels, channel_names = load_deap_data(file_path, selected_channels)

        # Extract features for each trial
        X = []
        for i in range(data.shape[0]):
            eeg_features = extract_eeg_features(data[i], selected_channels, channel_names, fs=fs)
            X.append(eeg_features)

        X_list.append(np.array(X))
        y_list.append(labels[:, :2])  # Valence and Arousal

    # Combine data from all files
    X_combined = np.vstack(X_list)
    y_combined = np.vstack(y_list)

    return X_combined, y_combined


# Example usage
if __name__ == "__main__":
    file_paths = [f'/Users/sudaxin/Desktop/data_preprocessed_python/s{str(i).zfill(2)}.dat' for i in range(1, 33)]
    selected_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']  # Example channels
    X, y = prepare_dataset(file_paths, selected_channels, fs=128)

    print("Combined features shape:", X.shape)
    print("Labels shape:", y.shape)
