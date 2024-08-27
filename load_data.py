import numpy as np
import pickle


def load_deap_data(file_path, selected_channels):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # Assuming DEAP data structure, update this list based on the actual channels in the DEAP dataset
    all_channel_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']  # Example channels

    # Identify indices of the selected channels
    channel_indices = [all_channel_names.index(ch) for ch in selected_channels]

    # Select only the data from the chosen channels
    selected_data = data['data'][:, channel_indices, :]

    # Assume that data['labels'] contains the labels (e.g., valence, arousal)
    labels = data['labels'][:, :2]  # Assuming the first two columns are valence and arousal

    return selected_data, labels, selected_channels


# Example usage
if __name__ == "__main__":
    # Create a list of file paths for all datasets
    file_paths = [f'/Users/sudaxin/Desktop/data_preprocessed_python/s{str(i).zfill(2)}.dat' for i in range(1, 33)]
    selected_channels = ['Fp1', 'Fp2']

    # Initialize lists to store data from all files
    X_list = []
    y_list = []

    # Load and prepare data from all datasets
    for file_path in file_paths:
        data, labels, channel_names = load_deap_data(file_path, selected_channels)
        X_list.append(data)
        y_list.append(labels)

    # Combine all datasets into a single array
    X = np.vstack(X_list)
    y = np.vstack(y_list)

    print("Combined Data shape:", X.shape)
    print("Combined Labels shape:", y.shape)
    print("Selected Channel names:", channel_names)
