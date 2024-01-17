import argparse
from mlflow.tracking import MlflowClient
from azureml.core import Run
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from surprise import KNNBasic
from surprise import accuracy
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
load_dotenv()

run = Run.get_context()
raw_movielens = run.input_datasets["movielens"]

parser = argparse.ArgumentParser("knn_user_rec")
parser.add_argument("--number_of_clusters", type=str, help="number of k in the knn algorithm")

args = parser.parse_args()

movielens = raw_movielens.to_pandas_dataframe()

# load data into a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(movielens, reader)

# split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize a MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
with mlflow.start_run() as run:
  params = {
    'k': args.number_of_clusters
  }
  algo = KNNBasic(
    k=params['k'],
    sim_options={'name': 'pearson_baseline', 'user_based': False})
  algo.fit(trainset)

  # run the trained model against the test-set
  test_pred = algo.test(testset)

  # get RMSE
  acc = accuracy.rmse(test_pred, verbose=True)
  
  # log parameters and metrics
  mlflow.log_params(params)
  mlflow.log_params(sim_options)
  mlflow.log_metric("test_rmse", acc)
  mlflow.sklearn.log_model(
      sk_model=algo,
      artifact_path=ARTIFACT_PATH,
      registered_model_name=MODEL_NAME, # Note: This will create a new registered model if it doesn't exist
  )
  mlflow.end_run()