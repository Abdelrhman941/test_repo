import mlflow, dagshub
dagshub.init(repo_owner='Abdelrhman941', repo_name='test_repo', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)