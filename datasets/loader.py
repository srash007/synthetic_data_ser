import pandas as pd

def load_dataset(name):
    """
    Load a dataset by name.
    
    Parameters:
    - name (str): The name of the dataset to load.
    
    Returns:
    - pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if name == "bank8fm":
        # Load bank8fm dataset
        return pd.read_csv("bank8fm.csv")
    elif name == "dataset2":
        # Load dataset2
        return pd.read_csv("path/to/dataset2.csv")
    else:
        raise ValueError(f"Dataset {name} not recognized.")