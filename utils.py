import io
import hashlib
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from ecdsa import SigningKey, SECP256k1
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import socket
import pickle


def getPrivate():
    return SigningKey.generate(curve=SECP256k1)

def getPublic(privateKey):
    return privateKey.verifying_key

def getWeightsHash(model): 
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    weights_bytes = buffer.getvalue()
    hash = hashlib.sha256(weights_bytes).hexdigest()
    return hash

def setOptimizer(model, optimizer, lr):
    if optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=lr)

def setCriterion(criterion):
    if criterion == "BCELoss":
        return nn.BCELoss()
    else:
        print("Criterion doesn't exist!")
    
def prepareData(n_samples=1000, test_size=0.3, n_trainers=1):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loaders = []
    for i in range(n_trainers):
        start_idx = i * len(X_test) // n_trainers
        end_idx = (i + 1) * len(X_test) // n_trainers if i < n_trainers - 1 else len(X_test)
        X_test_subset = X_test[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        X_test_subset_tensor = torch.tensor(X_test_subset, dtype=torch.float32)
        y_test_subset_tensor = torch.tensor(y_test_subset, dtype=torch.float32)
        test_subset_dataset = TensorDataset(X_test_subset_tensor, y_test_subset_tensor)
        test_subset_loader = DataLoader(test_subset_dataset, batch_size=16, shuffle=False)
        test_loaders.append(test_subset_loader)
    return train_loader, test_loaders

def save_data(file_path, data):
    with open(file_path, 'wb') as f:
        torch.save(data, f)

def prepare_data(n_samples=1000, test_size=0.3, n_trainers=1, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate dataset
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=1)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Convert training data to tensors and save them
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    save_data(os.path.join(data_dir, 'train_data.pth'), (X_train_tensor, y_train_tensor))
    
    # Prepare testing subsets, convert to tensors, and save them
    test_files = []
    for i in range(n_trainers):
        start_idx = i * len(X_test) // n_trainers
        end_idx = (i + 1) * len(X_test) // n_trainers if i < n_trainers - 1 else len(X_test)
        X_test_subset = X_test[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        
        X_test_subset_tensor = torch.tensor(X_test_subset, dtype=torch.float32)
        y_test_subset_tensor = torch.tensor(y_test_subset, dtype=torch.float32)
        
        file_path = os.path.join(data_dir, f'test_data_{i}.pth')
        save_data(file_path, (X_test_subset_tensor, y_test_subset_tensor))
        test_files.append(file_path)
    print(test_files)
    return test_files

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = torch.load(f)
    return data

def load_train_data(train_file):
    X_train_tensor, y_train_tensor = load_data(train_file)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    return train_loader

def load_test_data(test_file):
    X_test_subset_tensor, y_test_subset_tensor = load_data(test_file)
    test_subset_dataset = TensorDataset(X_test_subset_tensor, y_test_subset_tensor)
    test_subset_loader = DataLoader(test_subset_dataset, batch_size=16, shuffle=False)
    return test_subset_loader

def broadcast(ip_addresses, sequence):
    PORT = 5000
    serialized_sequence = pickle.dumps(sequence)  
    for ip_address in ip_addresses:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((ip_address, PORT))
            sock.sendall(len(serialized_sequence).to_bytes(4, byteorder='big'))
            sock.sendall(serialized_sequence)
            print(f"Sent sequence to {ip_address}")
        except Exception as e:
            print(f"Error sending to {ip_address}: {e}")
        finally:
            sock.close()

def handle_incoming_sequences(conn, addr, node):
    print(f"Connected by {addr}")
    try:
        data_size = int.from_bytes(conn.recv(4), byteorder='big')
        received_data = b''
        while len(received_data) < data_size:
            packet = conn.recv(4096)
            if not packet:
                break
            received_data += packet
        sequence = pickle.loads(received_data)
        node.queue.append(sequence)
        if len(node.queue) == 5:
            node.syncLedger()
        return sequence
        
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None
    finally:
        conn.close()

def start_listening(node):
    PORT = 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', PORT))
        s.listen()
        print(f"Listening on port {PORT}")
        while True:
            conn, addr = s.accept()
            handle_incoming_sequences(conn, addr, node)

def countZeros(hash_value):
    count = 0
    for char in hash_value:
        if char == '0':
            count += 1
        else:
            break
    return count

def proof_of_work(hash, difficulty):
    prefix = '0' * difficulty
    nonce = 0

    while True:
        data = f"{hash}{nonce}".encode('utf-8')
        hash_result = hashlib.sha256(data).hexdigest()
        if hash_result.startswith(prefix):
            return nonce, hash_result
        nonce += 1

class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.layer1 = nn.Linear(input_shape[0], 8, bias=True)
        self.layer2 = nn.Linear(8, 16, bias=True)
        self.layer3 = nn.Linear(16, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x