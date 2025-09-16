# test_repo
This repository is a remote repository for testing purposes:

---
1. add files
2. commit & push
3. go to [**dagshub**](https://dagshub.com/) and login and track experiments
4. go to create , New Repo, connect at repository, and connect to your remote repo don't select all repos
5. install then select the repo you want to track , connect repo. [*`now you are synchronized github with dagshub and can track experiments`*]

---
- create a new experiment in dagshub:
1. go to remote, experiments, copy the command like this:
```py
import dagshub
dagshub.init(repo_owner='Abdelrhman941', repo_name='test_repo', mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```
2. run the code in your local repo
```terminal
python test.py
```
3. go to dagshub and refresh the page you will see the experiment tracked in `Experiments` tab
4. so run your model script like ([1.logistic_script.py](files/1.logistic_script.py)) that contains the `dagshub.init` code
```python
import dagshub
dagshub.init(repo_owner='Abdelrhman941', repo_name='test_repo', mlflow=True)
```