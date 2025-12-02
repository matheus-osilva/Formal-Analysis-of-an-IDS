import pandas as pd
import numpy as np
import torch
import torch.onnx
import random
import os

from torch import nn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from neural_networks_training.utils import extractAllSets
from neural_networks_training.utils import z_scale
from neural_networks_training.utils import train_network_0
from neural_networks_training.utils import train_network_1
from neural_networks_training.neural_network import FFN

def setup_reproducibility(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set as: {seed}")

# Neural network characteristics
N_HL = 16  # Number of neurons in the hidden layers
SEED = 4221465
N_FEATURES = 5
N_OUTPUTS = 15
EPOCHS = 1000
LEARNING_RATE = 0.001
NN_FILE_NAME_0 = 'nn_0.torch'
NN_FILE_NAME_1 = 'nn_1.torch'
LABEL_COLUMN = 'Label'

setup_reproducibility(SEED)
    
# Loading the datasets into dataframes
df_data_1 = pd.read_parquet('CICIDS-2017_cleaned/Benign-Monday-no-metadata.parquet')
df_data_2 = pd.read_parquet('CICIDS-2017_cleaned/Botnet-Friday-no-metadata.parquet')
df_data_3 = pd.read_parquet('CICIDS-2017_cleaned/Bruteforce-Tuesday-no-metadata.parquet')
df_data_4 = pd.read_parquet('CICIDS-2017_cleaned/DDoS-Friday-no-metadata.parquet')
df_data_5 = pd.read_parquet('CICIDS-2017_cleaned/DoS-Wednesday-no-metadata.parquet')
df_data_6 = pd.read_parquet('CICIDS-2017_cleaned/Infiltration-Thursday-no-metadata.parquet')
df_data_7 = pd.read_parquet('CICIDS-2017_cleaned/Portscan-Friday-no-metadata.parquet')
df_data_8 = pd.read_parquet('CICIDS-2017_cleaned/WebAttacks-Thursday-no-metadata.parquet')

# Concatenating the dataframes to single dataframe
df_data = pd.concat([df_data_1, df_data_2, df_data_3, df_data_4, df_data_5, df_data_6, df_data_7, df_data_8], axis=0, ignore_index=True)

# Data preprocessing

null_counts = df_data.isnull().sum() # Find null values
print(f"{null_counts.sum()} null entries have been found in the dataset\n")
df_data.dropna(inplace=True) # Drop null values
duplicate_count = df_data.duplicated().sum() # Find duplicates
print(f"{duplicate_count} duplicate entries have been found in the dataset\n")
df_data.drop_duplicates(inplace=True)  # Remove duplicates
print(f"All duplicates have been removed\n")


df_data.reset_index(drop=True, inplace=True)

# Inspect the dataset for categorical columns
print("Categorical columns:",df_data.select_dtypes(include=['object']).columns.tolist(),'\n')

# Print the first 5 lines
df_data.head()

# Inspection of Target Feature
print('Shape of Dataframe: ',df_data.shape,'\n')
print('Inspection of Target Feature - y:\n')
df_data.Label.value_counts()

# Extract and normalize features as X
X = df_data.copy()
X = X.drop('Label', axis=1)
print(X.columns)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Extract and enumerate target as y
y = df_data.copy()
y = y['Label']
LABEL_CLASSES = y.unique()
label_map = {label: i for i, label in enumerate(LABEL_CLASSES)}
y = y.map(label_map)
y = pd.DataFrame(y)

# Selects top 5 features based on ANOVA's F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)
X_new_df = pd.DataFrame(X_new)
selected_features_mask = selector.get_support()
selected_column_names = X.columns[selected_features_mask]
print(selected_column_names)

# Extract all sets with percentages: 80-15-05 (%)
X_train, X_val, X_test, y_train, y_val, y_test = extractAllSets(X_new_df,y,0.80,0.15,0.5,SEED)

# Extract scaled sets
X_train_z, X_val_z, X_test_z = z_scale(X_train, X_val, X_test)
# Organize all sets in list format
scaled_data = [X_train_z, X_val_z, X_test_z, y_train, y_val, y_test]
# Organize all sets in list format
original_data = [X_train, X_val, X_test, y_train, y_val, y_test]

# --- Convert to PyTorch Tensors ---
X_train = torch.from_numpy(X_train.to_numpy()).float()
# y_train needs to be a LongTensor for CrossEntropyLoss
y_train = torch.from_numpy(y_train.to_numpy()).long() 
X_val = torch.from_numpy(X_val.to_numpy()).float()
y_val = torch.from_numpy(y_val.to_numpy()).long()
y_train = y_train.squeeze()
y_val = y_val.squeeze()

splitted_data = (X_train, y_train, X_val, y_val)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

# Instantiate the model 
network = FFN(n_features=N_FEATURES, n_hl=N_HL, n_outputs=N_OUTPUTS)

# Define the optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

# Define the loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Start the training process for neural network 0
network_0 = train_network_0(splitted_data, network, optimizer, criterion, NN_FILE_NAME_0, EPOCHS)

# Save as ONNX
network_0.eval()
toOnnxInput = torch.as_tensor([0]*N_FEATURES).float()
torch.onnx.export(network_0, toOnnxInput, "onnx/nn_0.onnx")

# Start the training process for neural network 1

# Resets the seed
setup_reproducibility(SEED) 

# Reinstantiate the model
network = FFN(n_features=N_FEATURES, n_hl=N_HL, n_outputs=N_OUTPUTS) 

LEARNING_RATE = 0.001
# Recreates optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

BATCH_SIZE = 64 
# Creates tensordatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Creates Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

network_1 = train_network_1(
    train_loader, 
    val_loader, 
    network, 
    optimizer, 
    criterion, 
    NN_FILE_NAME_1,
    EPOCHS,
    patience=15 # If the network doesn't improve in 15 epochs, trigger early stop
)

# Save as ONNX
network_1.eval()
toOnnxInput_1 = torch.as_tensor([0]*N_FEATURES).float()
torch.onnx.export(network_1, toOnnxInput_1, "onnx/nn_1.onnx")

# Set test data
X_test_tensor = torch.from_numpy(X_test_z.to_numpy()).float()
y_test_tensor = torch.from_numpy(y_test.to_numpy()).long().squeeze()

network_0.eval()
network_1.eval()

with torch.no_grad():
    logits_0 = network_0(X_test_tensor)
    logits_1 = network_1(X_test_tensor)
    
    predictions_0 = torch.argmax(logits_0, dim=1)
    predictions_1 = torch.argmax(logits_1, dim=1)

preds_numpy = predictions_0.numpy()
y_test_numpy = y_test_tensor.numpy()

acc = accuracy_score(y_test_numpy, preds_numpy)
print(f"Accuracy Score in Test Set for neural network 0: {acc * 100:.2f}%")

print("\nClassification Report for neural network 0:")
print(classification_report(y_test_numpy, preds_numpy))

preds_numpy = predictions_1.numpy()
y_test_numpy = y_test_tensor.numpy()

acc = accuracy_score(y_test_numpy, preds_numpy)
print(f"Accuracy Score in Test Set for neural network 1: {acc * 100:.2f}%")

print("\nClassification Report for neural network 0:")
print(classification_report(y_test_numpy, preds_numpy))
