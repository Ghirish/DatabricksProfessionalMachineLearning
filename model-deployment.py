# Import necessary libraries
import mlflow
import mlflow.spark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/dbfs/path/to/dataset.csv')
df = df[["Pclass", "Age", "Fare", "Survived"]].fillna(0)
spark_df = spark.createDataFrame(df)

X = df[["Pclass", "Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Train the Model and Register it in MLflow
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "random_forest_model")
    model_uri = "runs:/{}/random_forest_model".format(run.info.run_id)
    result = mlflow.register_model(model_uri, "RandomForestDeploymentModel")

# Step 2: Batch Deployment

model = mlflow.sklearn.load_model(model_uri)

# Create a Spark UDF to use the model for batch prediction
@udf(DoubleType())
def predict_batch(pclass, age, fare):
    return float(model.predict([[pclass, age, fare]])[0])

# Apply the UDF to the Spark DataFrame to make predictions
spark_df = spark_df.withColumn("prediction", predict_batch(col("Pclass"), col("Age"), col("Fare")))
spark_df.select("Pclass", "Age", "Fare", "prediction").show()

# Step 3: Streaming Deployment

# Convert the batch pipeline into a streaming pipeline

# Let's simulate incoming streaming data using a rate source in Spark
streaming_data = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
streaming_df = streaming_data.selectExpr("mod(value, 3) as Pclass", "value as Age", "(value * 10) as Fare")

# Apply the prediction UDF to streaming data
streaming_predictions = streaming_df.withColumn("prediction", predict_batch(col("Pclass"), col("Age"), col("Fare")))
query = streaming_predictions.writeStream.format("console").start()

import time
time.sleep(10)
query.stop()

# Step 4: Real-time Inference using Model Serving

# Assume we have the model serving enabled in Databricks for real-time prediction

# Load the model from the production stage (for real-time API calls)
model_uri = "models:/RandomForestDeploymentModel/Production"
production_model = mlflow.sklearn.load_model(model_uri)

# Real-time API simulation (you can replace this with actual REST API server setup)
def predict_real_time(pclass, age, fare):
    input_data = np.array([[pclass, age, fare]])
    prediction = production_model.predict(input_data)
    return prediction[0]

real_time_input = [1, 30, 200] 
real_time_prediction = predict_real_time(*real_time_input)

print(f"Real-time prediction: {real_time_prediction}")
