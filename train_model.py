# train_model.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from model_utils import NYCTaxiExampleDataset, MLP, train_model
from data_utils import raw_taxi_df, clean_taxi_df, split_taxi_data
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def main(filename: str, train_size: int, epochs: int, learning_rate: float):
    """
    Main function to load data, process it, and train the model.

    Args:
        filename (str): Path to the Parquet file containing the taxi data.
        train_size (int): Number of samples to use for training.
        epochs (int): Number of epochs for model training.
        learning_rate (float): Learning rate for the optimizer.

    The function performs the following steps:
    1. Loads the data from the Parquet file.
    2. Cleans the data (removes NaNs and outliers).
    3. Splits the data into training and test sets.
    4. Prepares the training data using the NYCTaxiExampleDataset class.
    5. Initializes the MLP model.
    6. Defines the loss function and optimizer.
    7. Trains the model using the specified parameters.
    """
    
    # Load and clean data
    raw_df = raw_taxi_df(filename=filename)
    print("Data loaded.")
    
    clean_df = clean_taxi_df(raw_df=raw_df)
    print("Data cleaned.")
    
    # Define feature columns and target column
    location_ids = ['PULocationID', 'DOLocationID']  # Using these columns for simplicity in this example
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                       x_columns=location_ids, 
                                                       y_column="fare_amount", 
                                                       train_size=train_size)
    print("Data split into training and test sets.")
    
    # Prepare dataset and DataLoader
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    print("Data prepared for model training.")
    
    # Initialize the MLP model
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
    print("Model initialized.")
    
    # Define loss function and optimizer
    loss_function = nn.L1Loss()  # L1 loss is suitable for regression problems
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(mlp, trainloader, optimizer, loss_function, epochs=epochs)

if __name__ == "__main__":
    # Define parameters for training
    FILENAME = "yellow_tripdata_2024-01.parquet"  # Change this to your actual file path
    TRAIN_SIZE = 500000  # Use 500,000 samples for training
    EPOCHS = 5  # Number of epochs
    LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer
    
    # Call the main function with these parameters
    main(filename=FILENAME, train_size=TRAIN_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)
