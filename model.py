"""Simple reproducible Iris classification project."""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data():
    """Load the built-in Iris dataset."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y, iris.target_names


def split_data(X, y, test_size=0.2):
    """Create reproducible stratified train/test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def scale_data(X_train, X_test):
    """Standardize predictors using training-set statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """Train a logistic-regression classifier."""
    model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return predictions and common classification metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    return predictions, accuracy, matrix, report


def run_pipeline():
    """Run the complete experiment from loading to evaluation."""
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    predictions, accuracy, matrix, report = evaluate_model(
        model, X_test_scaled, y_test
    )
    return {
        "target_names": target_names,
        "model": model,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Classification report:")
    print(results["classification_report"])
