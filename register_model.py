import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
experiment = client.get_experiment_by_name("Wine_Classification")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"]
)
if not runs:
    raise ValueError("No trained model runs found")
best_run = runs[0]
print("Best model run ID:", best_run.info.run_id)
print("Best F1 Score:", best_run.data.metrics["f1_score"])
model_uri = f"runs:/{best_run.info.run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="wine-classifier"
)
print("Model registered successfully")
print("Model version:", registered_model.version)