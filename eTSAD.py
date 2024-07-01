import os
import numpy as np

# Function to normalize the time series data
def z_score_normalize(time_series):
    return (time_series - np.mean(time_series, axis=0)) / np.std(time_series, axis=0)

# Function to create edge time series (eTS) from the normalized regional time series
def calculate_eTS(normalized_data):
    T, N = normalized_data.shape
    u, v = np.triu_indices(N, k=1)
    M = len(u)
    eTS = np.zeros((T, M))
    for i in range(M):
        eTS[:, i] = normalized_data[:, u[i]] * normalized_data[:, v[i]]
    return eTS, u, v

# Function to calculate edge functional connectivity (eFC)
def calculate_eFC(eTS):
    b = np.dot(eTS.T, eTS)  # inter product
    c = np.sqrt(np.diag(b))  # square root of variance
    d = np.outer(c, c)      # normalization matrix
    eFC = b / d             # divide to get correlation
    return eFC

# Input and output directories
input_dir = '/home/djyang/AAL90'
output_eTS_dir = '/home/djyang/EdgeC/eTS_AD'
output_eFC_dir = '/home/djyang/EdgeC/eFC_AD'
os.makedirs(output_eTS_dir, exist_ok=True)
os.makedirs(output_eFC_dir, exist_ok=True)

# Traverse the input directory
for file in os.listdir(input_dir):
    if file.endswith('.txt'):
        file_path = os.path.join(input_dir, file)
        
        # Load the .txt file
        time_series_data = np.loadtxt(file_path)

        # Normalize the time series for each region
        normalized_data = z_score_normalize(time_series_data)

        # Create the edge time series (eTS)
        eTS, u, v = calculate_eTS(normalized_data)

        # Calculate the edge functional connectivity (eFC)
        eFC = calculate_eFC(eTS)

        # Define the output file paths
        base_name = os.path.splitext(file)[0]
        eTS_output_file_path = os.path.join(output_eTS_dir, f'{base_name}_eTS.npy')
        eFC_output_file_path = os.path.join(output_eFC_dir, f'{base_name}_eFC.npy')

        # Save the eTS and eFC matrices to .npy files
        np.save(eTS_output_file_path, eTS)
        np.save(eFC_output_file_path, eFC)

        print(f"eTS calculation complete for {file_path}. Results saved to {eTS_output_file_path} and {eFC_output_file_path}")
