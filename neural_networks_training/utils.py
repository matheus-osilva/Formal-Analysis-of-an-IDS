import pandas as pd
import torch
import copy

# for data preparation
from sklearn.model_selection import train_test_split

# for scaling
from sklearn.preprocessing import StandardScaler

SEED = 314159

def extractAllSets(X,y,p_train,p_val,p_test,random_state=SEED, shuffle=True):
    ''''
    Splits a given pandas dataframe X (features) into three subsets:X_train, X_val and X_test. 
    Also splits a given pandas series y (target) to y_train, y_val and y_test respectively. 
    Fractional ratios are provided by the user, as percentages, namely: p_train, p_valid, p_test. 
    These inputs describe the percentage of the extracted sets in reference with the inputs X and y. 
    The final sets are extracted by executing method train_test_split() twice.

    Parameters
    ----------
    X: pandas Dataframe
    y: pandas Series
    p_train, p_val, p_test  : float
    random_state : integer
    shuffle: (boolean) Enables shuffling the dataset
    
    The values should be expressed as float fractions and  should  sum to 1.0.
    The parameter of random_state ensures reproducibility.
    
    Returns
    -------
    X_train, X_val, X_test :
        Dataframes (features) containing the three splits.
    y_train, y_val, y_test  :
        Series (targets) containig the three splits
    '''
    # The initial train-test split produces X_train and y_train
    # Two additional sets are created X_temp and y_temp, that will produce the rest of the sets
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,                
                                                        stratify=y,
                                                        test_size=(1.0 - p_train),
                                                        random_state=random_state,
                                                        shuffle=shuffle)  # Enable/desable shuffling
    # Note that by applying the stratify condition we ensure homogeneous distribution 
    # of chareacteristics in targets
    
    # Parameter fraction describes the relevant size of the test size
    fraction = p_test / (p_val + p_test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,stratify=y_temp,
                                                      test_size=fraction,
                                                      random_state=random_state,
                                                      shuffle=shuffle)  # Enable/desable shuff
    # Note that we use the same random_state twice for reproducibility
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_accuracy(y_true, y_pred):
    """
    Calculates accuracy for a multi-class classification task.
    
    Args:
        y_true (torch.Tensor): The true labels (class indices). Shape: [batch_size]
        y_pred (torch.Tensor): The raw model predictions (logits). Shape: [batch_size, n_classes]
        
    Returns:
        float: The accuracy score.
    """
    # Get the predicted class index by finding the max logit
    predicted = torch.argmax(y_pred, dim=1)
    # Count how many predictions match the true labels
    correct_predictions = (predicted == y_true).sum().float()
    # Calculate accuracy
    accuracy = correct_predictions / len(y_true)
    return accuracy.item()

def round_tensor(t, decimal_places=4):
    """Utility function to round a tensor's item for printing."""
    return round(t.item(), decimal_places)

def train_network_0(data, network, optimizer, criterion, nn_file_name, EPOCHS):
    """
    Main training loop for the neural network.
    """
    (x_train, y_train, x_val, y_val) = data

    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        # Set the model to training mode
        network.train()
        
        # Forward pass
        y_pred = network(x_train)
        
        # Calculate loss
        train_loss = criterion(y_pred, y_train)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            # Set the model to evaluation mode for inference
            network.eval()
            with torch.no_grad(): # Disable gradient calculation for validation
                train_acc = calculate_accuracy(y_train, y_pred)
                
                # Validation pass
                y_val_pred = network(x_val)
                val_loss = criterion(y_val_pred, y_val)
                val_acc = calculate_accuracy(y_val, y_val_pred)

                tr_lss = round_tensor(train_loss)
                vl_lss = round_tensor(val_loss)
                
                print(f"Epoch {epoch:04d}: Train Loss: {tr_lss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {vl_lss:.4f}, Val Acc: {val_acc:.4f}")

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    print("--- Training Finished ---")
    # Save the trained model
    torch.save(network.state_dict(), nn_file_name)
    print(f"Model saved to {nn_file_name}")

    return network

def train_network_1(train_loader, val_loader, network, optimizer, criterion, nn_file_name, EPOCHS, patience=10):
    print("--- Starting Training (Mini-Batch) ---")
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(network.state_dict())
    patience_counter = 0 # Counter for early stop
    
    for epoch in range(EPOCHS):
        
        # --- TRAINING PHASE ---
        network.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            # 1. Set gradients to zero
            optimizer.zero_grad()
            
            # 2. Forward Pass
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            
            # 3. Backward Pass
            loss.backward()
            optimizer.step()
            
            # Batch statistics
            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            running_corrects += (predictions == labels).sum().item()
            total_samples += inputs.size(0)
            
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples

        # --- VALIDATION PHASE ---
        network.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs, dim=1)
                val_running_corrects += (predictions == labels).sum().item()
                val_samples += inputs.size(0)
        
        epoch_val_loss = val_running_loss / val_samples
        epoch_val_acc = val_running_corrects / val_samples

        # --- LOGGING AND EARLY STOPPING ---
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Checks if its improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(network.state_dict())
            torch.save(network.state_dict(), nn_file_name) # Saves on drive
            patience_counter = 0 # Resets patience
            print(f"--> Model saved! (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early Stopping! Stopping training.")
            break

    print("--- Training Finished ---")
    # Loads best model
    network.load_state_dict(best_model_wts)
    return network
