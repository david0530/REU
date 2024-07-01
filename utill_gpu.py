import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from scipy.sparse import csr_matrix, diags
import scipy.io
from models import GCN
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the device to GPU 3
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def load_edge_data(edge_filepath):
    edge_data = torch.load(edge_filepath)
    eFC = edge_data['eFC']
    labels = edge_data['label']
    if isinstance(labels, int):  # If labels are mistakenly saved as a single integer
        labels = [labels]  # Convert to list for consistency
    return eFC, labels

def load_node_data(node_filepath):
    mat = scipy.io.loadmat(node_filepath)
    node_features = mat['data']  # Adjust according to the actual key in the .mat file
    return node_features

def create_transition_matrix(n_nodes, u, v):
    T = np.zeros((n_nodes, len(u)), dtype=int)
    for i, (u_node, v_node) in enumerate(zip(u, v)):
        T[u_node, i] = 1
        T[v_node, i] = 1
    return T

def load_edge_features(edge_feature_path):
    edge_features = np.load(edge_feature_path)
    edge_features_mean = np.mean(edge_features, axis=0).reshape(-1, 1)
    return torch.tensor(edge_features_mean, dtype=torch.float32).to(device)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)).to(device)
    values = torch.from_numpy(sparse_mx.data).to(device)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.where(rowsum == 0, 0., 1. / rowsum)
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_id_from_filename(filename):
    import re
    pattern = r'sub-([a-zA-Z0-9]+)'  # Adjusted pattern to capture alphanumeric IDs
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        raise ValueError("Filename format does not match the expected pattern.")

def create_fully_connected_adj(n_nodes):
    adj = np.ones((n_nodes, n_nodes), dtype=int)
    np.fill_diagonal(adj, 0)
    return adj

def search_node_files(root_dir):
    node_files = []
    for subdir, _, files in tqdm(os.walk(root_dir), desc="Scanning node directories"):
        if os.path.basename(subdir).startswith('sub-'):
            for file in files:
                if file.endswith('AAL116_correlation_matrix.mat'):
                    file_path = os.path.join(subdir, file)
                    node_files.append(file_path)
    return node_files

def prepare_data(edge_files, node_files, edge_feature_files, edge_dir, node_dir, edge_feature_dir):
    data = []
    for edge_file in edge_files:
        edge_id = get_id_from_filename(edge_file)
        node_file = next((f for f in node_files if get_id_from_filename(f) == edge_id), None)
        edge_feature_file = next((f for f in edge_feature_files if get_id_from_filename(f) == edge_id), None)

        if node_file and edge_feature_file:
            edge_filepath = os.path.join(edge_dir, edge_file)
            node_filepath = os.path.join(node_dir, node_file)
            edge_feature_path = os.path.join(edge_feature_dir, edge_feature_file)

            eadj, labels = load_edge_data(edge_filepath)
            node_features = load_node_data(node_filepath)
            edge_features = load_edge_features(edge_feature_path)

            eadj = sparse_mx_to_torch_sparse_tensor(csr_matrix(eadj)).to(device)
            n_nodes = node_features.shape[0]
            node_adj = create_fully_connected_adj(n_nodes)
            node_adj = sparse_mx_to_torch_sparse_tensor(csr_matrix(node_adj)).to(device)
            u, v = np.triu_indices(n_nodes, k=1)
            transition_matrix = create_transition_matrix(n_nodes, u, v)
            transition_matrix = torch.FloatTensor(transition_matrix).to(device)
            normalized_features = normalize(csr_matrix(node_features))
            normalized_features = torch.FloatTensor(normalized_features.todense()).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
            data.append((normalized_features, edge_features.to(device), eadj, node_adj, transition_matrix, labels_tensor))
    return data

def train(model, optimizer, criterion, train_data, log_file, epoch):
    model.train()
    with open(log_file, 'a') as f:
        for data in train_data:
            normalized_features, edge_features, eadj, node_adj, transition_matrix, labels_tensor = data
            batch_indices = torch.zeros(len(labels_tensor), dtype=torch.long).to(device)  # Assuming single batch for simplicity

            # Forward pass
            outputs = model(normalized_features, edge_features, eadj, node_adj, transition_matrix, batch_indices)
            loss = criterion(outputs, labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted_labels = torch.max(outputs, 1)
            f.write(f"Epoch {epoch} - Training - Loss: {loss.item()}\n")
            f.write(f"Epoch {epoch} - Training - Actual Labels: {labels_tensor.tolist()}\n")
            f.write(f"Epoch {epoch} - Training - Predicted Labels: {predicted_labels.tolist()}\n")

def test(model, criterion, test_data, log_file, epoch):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    with open(log_file, 'a') as f:
        with torch.no_grad():
            for data in test_data:
                normalized_features, edge_features, eadj, node_adj, transition_matrix, labels_tensor = data
                batch_indices = torch.zeros(len(labels_tensor), dtype=torch.long).to(device)  # Assuming single batch for simplicity

                # Forward pass
                outputs = model(normalized_features, edge_features, eadj, node_adj, transition_matrix, batch_indices)
                loss = criterion(outputs, labels_tensor)
                total_loss += loss.item()

                _, predicted_labels = torch.max(outputs, 1)
                all_labels.extend(labels_tensor.tolist())
                all_preds.extend(predicted_labels.tolist())

                f.write(f"Epoch {epoch} - Testing - Loss: {loss.item()}\n")
                f.write(f"Epoch {epoch} - Testing - Actual Labels: {labels_tensor.tolist()}\n")
                f.write(f"Epoch {epoch} - Testing - Predicted Labels: {predicted_labels.tolist()}\n")

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(test_data)
    return accuracy, avg_loss

def main():
    edge_dir = '/home/djyang/EdgeC/PPMI_eFC_data_mini'
    node_dir = '/home/djyang/ppmi'
    edge_feature_dir = '/home/djyang/EdgeC/eTC'

    edge_files = [f for f in os.listdir(edge_dir) if f.endswith('.pt')]
    node_files = search_node_files(node_dir)
    edge_feature_files = [f for f in os.listdir(edge_feature_dir) if f.endswith('.npy')]

    # Initialize the GCN model and optimizer
    nfeat_v = 116  # Adjust this based on your node feature dimension
    nfeat_e = 1    # Adjust this based on your edge feature dimension
    nhid = 16
    nclass = 4     # Adjust according to the number of classes in your labels
    dropout = 0.5
    model = GCN(nfeat_v, nfeat_e, nhid, nclass, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Prepare data
    data = prepare_data(edge_files, node_files, edge_feature_files, edge_dir, node_dir, edge_feature_dir)

    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    log_file = 'gpu_training_testing_log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)  # Remove the log file if it already exists to start fresh

    # Training and testing phase
    num_epochs = 50  # Define the number of epochs
    for epoch in range(num_epochs):
        train(model, optimizer, criterion, train_data, log_file, epoch)
        accuracy, avg_loss = test(model, criterion, test_data, log_file, epoch)

        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch} - Average Loss: {avg_loss}\n")
            f.write(f"Epoch {epoch} - Accuracy: {accuracy}\n")

if __name__ == '__main__':
    main()
