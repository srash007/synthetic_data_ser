import pandas as pd
import smogn

class SMOGNPreprocessor:
    """
    Wrapper around the SMOGN algorithm.

    Parameters
    ----------
    rel_thres : float, default=0.8
        Relevance threshold defining rare observations.

    k : int, default=5
        Number of nearest neighbors.

    pert : float, default=0.02
        Gaussian noise perturbation.

    rel_method : str, default="auto"
        Method used to compute the relevance function.
    """

    def __init__(
        self,
        rel_thres=0.8,
        rel_method="auto",
        samp_method="balance",
        k=5,
        pert=0.02,
        
    ):

        self.rel_thres = rel_thres
        self.k = k
        self.pert = pert
        self.rel_method = rel_method
        self.samp_method = samp_method
        
    
    def fit_resample(self, X, y):
        """
        Apply SMOGN resampling.

        Parameters
        ----------
        X : ndarray or DataFrame
        y : ndarray or Series

        Returns
        -------
        X_resampled
        y_resampled
        """

        # Build DataFrame expected by smogn
        X = pd.DataFrame(X).copy()
        X["target"] = y

        # Apply SMOGN
        data_resampled = smogn.smoter(
            data=X,
            y="target",
            rel_thres=self.rel_thres,
            rel_method=self.rel_method,
            k=self.k,
            pert=self.pert,
            samp_method=self.samp_method 
            
        )

        # Split predictors and target
        X_resampled = data_resampled.drop(columns="target").to_numpy()
        y_resampled = data_resampled["target"].to_numpy()

        return X_resampled, y_resampled