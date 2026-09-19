import warnings
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

warnings.filterwarnings("ignore", category=FutureWarning)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLOps Model Versioning")

model_name = "Wine_Classification_Model"

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

client = MlflowClient()

models = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 150, "max_depth": 10}
]

versions = []

for params in models:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    signature = infer_signature(X_train, model.predict(X_train))

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=model_name
        )

        run_id = run.info.run_id

    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    current_version = max(
        int(v.version)
        for v in model_versions
        if v.run_id == run_id
    )

    versions.append({
        "version": current_version,
        "accuracy": accuracy,
        "f1_score": f1,
        "run_id": run_id
    })

    print(
        f"Version {current_version} registered | "
        f"Accuracy: {accuracy:.4f} | "
        f"F1 Score: {f1:.4f}"
    )

versions.sort(
    key=lambda x: (x["accuracy"], x["f1_score"]),
    reverse=True
)

best_version = versions[0]["version"]

print("\nModel Version Comparison")
print("-" * 50)

for version in versions:
    print(
        f"Version {version['version']} | "
        f"Accuracy: {version['accuracy']:.4f} | "
        f"F1 Score: {version['f1_score']:.4f}"
    )

print(f"\nBest model version: {best_version}")

for version in versions:
    model_version = version["version"]

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging"
    )

    print(f"Version {model_version} -> Staging")

client.transition_model_version_stage(
    name=model_name,
    version=best_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Version {best_version} -> Production")

for version in versions:
    model_version = version["version"]

    if model_version != best_version:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Archived"
        )

        print(f"Version {model_version} -> Archived")

print("\nFinal Model Registry Status")
print("-" * 50)

registered_versions = client.search_model_versions(
    f"name='{model_name}'"
)

for version in sorted(
    registered_versions,
    key=lambda x: int(x.version)
):
    print(
        f"Version {version.version} | "
        f"Stage: {version.current_stage} | "
        f"Run ID: {version.run_id}"
    )

print("\nBest model selected for deployment:")
print(f"Model: {model_name}")
print(f"Version: {best_version}")
print("Stage: Production")

print("\nProcess completed successfully.")