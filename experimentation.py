# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from mlflow.models.signature import infer_signature
import shap
import matplotlib.pyplot as plt

# 1. Data Management with Delta Table

df = pd.read_csv('/dbfs/path/to/dataset.csv') 
spark_df = spark.createDataFrame(df)

delta_path = "/tmp/delta_table"
spark_df.write.format("delta").mode("overwrite").save(delta_path)

delta_df = spark.read.format("delta").load(delta_path)

delta_df.createOrReplaceTempView("delta_table")
spark.sql("DESCRIBE HISTORY delta.`{}`".format(delta_path)).show()

version_df = spark.read.format("delta").option("versionAsOf", 1).load(delta_path)

delta_df = delta_df.select(col("Pclass"), col("Age"), col("Fare"), col("Survived"))
train_df = delta_df.toPandas()

# 2. Experiment Tracking with MLflow
X = train_df[["Pclass", "Age", "Fare"]].fillna(0)  # Feature columns
y = train_df["Survived"]  # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
mlflow.set_experiment("ML Certification Experiment")

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
  
    accuracy = accuracy_score(y_test, predictions)
  
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # 3. Advanced Experiment Tracking with Model Signatures and Input Examples
    input_example = np.array(X_test[:5])
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=input_example)
  
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
  
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("/tmp/shap_plot.png")
    plt.close()
    mlflow.log_artifact("/tmp/shap_plot.png")

# 4. Autologging and Hyperparameter Tuning with Hyperopt

mlflow.sklearn.autolog()

# Hyperparameter tuning with Hyperopt
def objective(params):
    with mlflow.start_run(nested=True):
        rf = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']))
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        return {'loss': -accuracy, 'status': STATUS_OK}

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 5, 20, 1)
}

# Start hyperparameter search with Hyperopt
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

print("Best Hyperparameters:", best_params)
