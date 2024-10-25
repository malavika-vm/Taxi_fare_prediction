
**NYC Taxi Fare Prediction**

This project trains a neural network (Multilayer Perceptron) to predict
taxi fares in New York City. It includes data preprocessing, cleaning,
splitting, and training a model based on location data.

**Dependencies**

All dependencies are listed in `requirements.txt`. To create this file,
use:

```{bash}
 pip freeze > requirements.txt
```

To install the dependencies, run:

```{bash}
pip install -r requirements.txt 
```

Main Dependencies

-   `pandas` for data manipulation

-   `scikit-learn` for preprocessing and splitting the data

-   `torch` (PyTorch) for building and training the neural network

-   `pyarrow` (for reading Parquet files)

    **How to Run the Project**

-   Step 1: Prepare the Data Place the dataset (e.g.,
    `yellow_tripdata_2024-01.parquet`) in the `data/` folder.

-   Step 2: Preprocess and Split the Data The script `train_model.py`
    will handle loading, cleaning, and splitting the data. It uses
    functions from `data_utils.py` to process the dataset. In this
    example, the pickup (PULocationID) and drop-off (DOLocationID)
    locations are used as features, but you can modify this in the
    `train_model.py` script to include more features.

-   Step 3: Train the Model To start training the model, navigate to the
    `scripts/` directory and run:

`python train_model.py`

You can configure the number of training samples, epochs, and learning
rate directly in the `train_model.py` script:

```{python}

FILENAME = "../data/yellow_tripdata_2024-01.parquet" # Path to the dataset 

TRAIN_SIZE = 500000 # Number of samples to use for training 

EPOCHS = 5 # Number of epochs 

LEARNING_RATE = 1e-4 #Learning rate for the Adam optimizer 

```

-   Step 4: Model Output The trained model can be saved to the `models/`
    directory if needed, and Jupyter notebooks for analysis can be
    stored in the `notebooks/` folder.
