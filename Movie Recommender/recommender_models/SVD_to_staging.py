"""
This file constains an MLFlow pipeline 
that takes the model in the model registry of
MLFlow and updates the model stage to staging.
NOTE: this updated the LATEST version of the model

# !pip install python-dotenv azure-ai-ml mlflow azureml-mlflow
"""

import warnings
warnings.filterwarnings("ignore")

import mlflow
import os
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
load_dotenv()


# Model parameters
SVD_EXPERIMENT_NAME = "SVD-Product-Based-Recommendation-Experiment"
SVD_MODEL_NAME = "SVD-Product-Based-Recommendation-Model"
SVD_ARTIFACT_PATH = "artifact-path-model"

# Set Azure variables and set mlflow tracking uri to Azure
if 'AZURE_CLIENT_ID' not in os.environ:
  raise Exception("It seems that Azure environment variables are not set. Please ask alex to provide you with the .env file.")

subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_WORKSPACE")

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

azureml_tracking_uri = ml_client.workspaces.get(
    ml_client.workspace_name
).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_tracking_uri)

###############################################################################

client = MlflowClient()

# get latest version
model_metadata = client.get_latest_versions(SVD_MODEL_NAME, stages=["None"])
latest_model_version = model_metadata[0].version

# transition model to staging
client.transition_model_version_stage(
    name=SVD_MODEL_NAME, 
    stage="Staging",
    version=latest_model_version
)