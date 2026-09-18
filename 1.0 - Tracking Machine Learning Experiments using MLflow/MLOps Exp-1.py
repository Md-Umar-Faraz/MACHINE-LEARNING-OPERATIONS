import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLflow Model Tracking")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

c_values = [0.1, 1.0, 10.0]

for c in c_values:
    with mlflow.start_run():
        model = LogisticRegression(C=c, max_iter=200)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("C", c)
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, name="model")

        print("C:", c)
        print("Accuracy:", accuracy)
        print("Run completed")
        print("-" * 30)