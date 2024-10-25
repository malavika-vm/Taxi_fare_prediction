# model_utils.py
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder

class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for NYC Taxi data.

    This class takes in the training feature and target DataFrames,
    applies one-hot encoding to the feature set, and prepares the data for PyTorch.

    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.DataFrame): Training target set.

    Attributes:
        X (torch.Tensor): Tensor of one-hot encoded features.
        y (torch.Tensor): Tensor of target values.
        X_enc_shape (int): Number of one-hot encoded features (input size for the model).
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray())  # One-hot encode X
        self.y = torch.from_numpy(self.y_train.values)  # Convert y to tensor
        self.X_enc_shape = self.X.shape[-1]  # Get the number of encoded features
        print(f"Encoded shape: {self.X_enc_shape}")
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        """
        Applies one-hot encoding to the feature DataFrame.

        Returns:
            scipy.sparse.csr.csr_matrix: One-hot encoded feature set.
        """
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """
    Defines a Multilayer Perceptron (MLP) for fare prediction.

    The MLP takes in one-hot encoded features and outputs a predicted fare amount.

    Args:
        encoded_shape (int): The number of input features (i.e., the number of one-hot encoded columns).
    
    Architecture:
        - Input layer: Encoded features.
        - Two hidden layers with ReLU activation.
        - Output layer with 1 unit (regression target).

    """
    
    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),  # Hidden layer 1
            nn.ReLU(),                     # Activation function
            nn.Linear(64, 32),             # Hidden layer 2
            nn.ReLU(),                     # Activation function
            nn.Linear(32, 1)               # Output layer (1 output for regression)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(mlp, trainloader, optimizer, loss_function, epochs=5):
    """
    Training loop for the MLP model.

    Args:
        mlp (MLP): The neural network model.
        trainloader (DataLoader): DataLoader object for the training data.
        optimizer (torch.optim.Optimizer): Optimizer to use for updating the model's weights.
        loss_function (torch.nn.modules.loss._Loss): Loss function to compute the error.
        epochs (int): Number of epochs to train the model.

    Trains the model by iterating through the training data, performing forward and backward passes,
    and updating the model's weights.
    """
    
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))  # Reshape targets
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0
    
    print('Training process has finished.')
