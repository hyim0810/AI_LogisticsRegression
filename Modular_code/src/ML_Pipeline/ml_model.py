from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from .feature_scaling import scale_features  # Import module scaling mới

def prepare_model(df, class_col, cols_to_exclude):
    """Prepare data for modeling with scaling"""
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference([class_col])]
    X = X[X.columns.difference(cols_to_exclude)]
    y = df[class_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Chạy model với dữ liệu đã scale
    model_log, y_pred = run_model(X_train_scaled, X_test_scaled, y_train, y_test)

    return model_log, y_pred, scaler

def run_model(X_train, X_test, y_train, y_test):
    """Run logistic regression with improved parameters"""
    logreg = LogisticRegression(
        random_state=13,
        max_iter=1000,  # Tăng số lượng iterations
        tol=1e-4,  # Có thể điều chỉnh tolerance
        solver='lbfgs',  # Vẫn giữ solver mặc định
        n_jobs=-1  # Sử dụng tất cả CPU cores
    )

    # Fit model
    logreg.fit(X_train, y_train)

    # Predict
    y_pred = logreg.predict(X_test)

    # Evaluate
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    print(classification_report(y_test, y_pred))
    print(f"The area under the curve is: {logit_roc_auc:.2f}")

    # Check convergence
    if not logreg.n_iter_ < logreg.max_iter:
        print("Warning: Model may not have converged. Consider:")
        print("1. Increasing max_iter")
        print("2. Adjusting tol parameter")
        print("3. Trying different solver")

    return logreg, y_pred
