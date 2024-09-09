# Import necessary libraries
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Step 1: Preprocessing Logic and Custom Model Classes
df = pd.read_csv('/dbfs/path/to/dataset.csv')
df = df[["Pclass", "Age", "Fare", "Survived"]].fillna(0)

X = df[["Pclass", "Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import sklearn
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = context.artifacts["model"]
    
    def predict(self, context, model_input):
        scaled_input = self.scaler.transform(model_input)
        return self.model.predict(scaled_input)

# Step 2: Training and Registering the Model

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, rf_model):
        self.rf_model = rf_model

    def predict(self, context, input_data):
        return self.rf_model.predict(input_data)

# Log model with MLflow
with mlflow.start_run() as run:
    # Log the model with preprocessing logic
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModelWrapper(model),
        registered_model_name="CustomModel"
    )

    # Log the scaler as well (if preprocessing is needed)
    mlflow.log_artifact("/path/to/scaler.pkl", artifact_path="preprocessing")

    # Register the model in MLflow Model Registry
    result = mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id),
        "CustomModel"
    )

# Step 3: Managing the Model Registry
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="CustomModel",
    version=result.version,
    stage="Staging"
)

client.update_model_version(
    name="CustomModel",
    version=result.version,
    description="Random Forest model with preprocessing logic"
)

# Step 4: Automating the Model Lifecycle with Jobs and Webhooks

# Define a webhook action when model is transitioned to 'Production'
webhook_payload = {
    "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
    "model_name": "CustomModel",
    "http_url_spec": {
        "url": "https://example.com/webhook",
        "authorization": "Bearer my-webhook-token"
    }
}

webhook = client.create_webhook(
    webhook_payload["model_name"],
    webhook_payload["http_url_spec"]
)

job_payload = {
    "name": "Deploy Model Job",
    "new_cluster": {
        "spark_version": "9.1.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
    },
    "notebook_task": {
        "notebook_path": "/Users/your_username/deploy_model"
    }
}

# Create a job for the model deployment
job_id = client.create_job(job_payload)

# Update the webhook to trigger this job when the model is transitioned to 'Production'
client.update_webhook(
    webhook_id=webhook.id,
    http_url_spec={
        "url": f"https://example.com/jobs/run-now?job_id={job_id}"
    }
)

