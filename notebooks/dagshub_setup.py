import dagshub
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/rakesh-kumar-22/emotion-detection-mlops.mlflow")
dagshub.init(repo_owner='rakesh-kumar-22', repo_name='emotion-detection-mlops', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)