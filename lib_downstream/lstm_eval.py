import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from parameter_parser import parameter_parser
import config
import os
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import roc_auc_score

#from https://www.researchgate.net/publication/361505045_A_Novel_Fully_Convolutional_Neural_Network_Approach_For_Detection_and_Classification_of_Attacks_on_Industrial_IoT_Devices_in_Smart_Manufacturing_Systems
#claim 98% accuracy 0.05 loss

# Define the CNN-LSTM model according to the updated structure where the dropout is placed before the LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, input_features, sequence_length):
        super(CNN_LSTM, self).__init__()

        # Convolutional 1D layer with ReLU activation
        self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=sequence_length, kernel_size=1)
        self.relu = nn.ReLU()
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(num_features=sequence_length)

        # Dropout layer applied before LSTM
        self.dropout = nn.Dropout(0.2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=sequence_length, hidden_size=8, num_layers=1, batch_first=True)

        # Dense (Fully Connected) layer with Sigmoid activation for binary classification
        self.dense = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):        
        # Apply convolutional layer
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        # Apply dropout as per the updated architecture
        x = self.dropout(x)

        # Prepare data for LSTM
        # We assume the input x is of shape (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # Rearrange dimensions to (batch_size, sequence_length, channels)
        
        # Apply LSTM
        x, (hn, cn) = self.lstm(x)

        # We only use the last output of the sequence from LSTM for the final prediction
        x = x[:, -1, :]  # Shape (batch_size, hidden_size)

        # Apply Dense layer with sigmoid activation
        x = self.dense(x)
        x = self.sigmoid(x)

        return x

# Initialize weights with Xavier Uniform initializer
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def eval(model, train_loader, test_loader):

    # Define the loss function (Binary Cross-Entropy)
    criterion = nn.BCELoss()

    # Define the optimizer (Adam with a learning rate of 0.03)
    optimizer = Adam(model.parameters(), lr=0.03)

    # Define the number of epochs
    epochs = 10

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for inputs, targets in train_loader:            
            # Forward pass
            #print(inputs.size())               
            #outputs = model(inputs.permute(1,0))
            outputs = model(inputs)        
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        all_predictions = []
        all_targets = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predictions = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                all_predictions.extend(predictions.flatten().tolist())
                all_targets.extend(targets.flatten().tolist())
                
        # Calculate metrics
        accuracy = sum([1 for i in range(len(all_predictions)) if (all_predictions[i] >= 0.5) == all_targets[i]]) / len(all_predictions)
        auroc = roc_auc_score(all_targets, all_predictions)
        print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}')



# Test driver function
def main(args):
    os.chdir("../../")
    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    file_prefix = args['dataset_name']

    datasets = {
        file_prefix + 'Raw': config.RAW_DATA_PATH + file_prefix + '.csv',
        file_prefix + 'Syn': config.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv'),
        file_prefix + 'Trivial': config.SYNTHESIZED_RECORDS_PATH + file_prefix +'_syn_trivial.csv'
    }

    # Load the dataset
    file_path = datasets[file_prefix + 'Raw']
    data = pd.read_csv(file_path)
    data = data.sort_values(by="ts")

    # Separate features and target label
    X = data.drop('label', axis=1)
    y = data['label']

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Apply One-Hot Encoding to categorical columns
    ohe = OneHotEncoder(sparse=False)
    X_categorical = ohe.fit_transform(X[categorical_cols])
    X_categorical = pd.DataFrame(X_categorical, columns=ohe.get_feature_names_out(categorical_cols))

    # Drop the original categorical columns and concatenate the one-hot encoded columns
    X.drop(categorical_cols, axis=1, inplace=True)
    X = pd.concat([X, X_categorical], axis=1)

    # Apply Robust Scaler to the features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Assume the dataset is already sorted by time. Now split the data without shuffling.
    #split_ratio = 0.8
    #split_index = int(len(X_scaled) * split_ratio)
    #X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    #y_train, y_test = y[:split_index], y[split_index:]

    #Split randomly
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(2)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(2)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Print the shapes of the processed datasets
    print(f'Train Features Shape: {X_train_tensor.shape}')
    print(f'Train Labels Shape: {y_train_tensor.shape}')
    print(f'Test Features Shape: {X_test_tensor.shape}')
    print(f'Test Labels Shape: {y_test_tensor.shape}')

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64  # You can set your batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_features = X_train_tensor.shape[1]

    model = CNN_LSTM(num_features, sequence_length=32)
    model.apply(init_weights)

    eval(model, train_loader, test_loader)

if __name__ == "__main__":
    args = parameter_parser()
    
    main(args)