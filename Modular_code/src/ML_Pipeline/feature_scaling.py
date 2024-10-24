from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler

    Parameters:
    -----------
    X_train : DataFrame or array-like
        Training data to be scaled
    X_test : DataFrame or array-like
        Test data to be scaled

    Returns:
    --------
    X_train_scaled : array-like
        Scaled training data
    X_test_scaled : array-like
        Scaled test data
    scaler : StandardScaler
        Fitted scaler object for future use
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame if input was DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler