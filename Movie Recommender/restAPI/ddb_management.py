import pandas as pd
import os
from io import StringIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, _container_client

class DDB_manager():
    def __init__(self) -> None:
        try:

            # Create the BlobServiceClient object
            self.blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))
            print("Setting up blob storage...")
            data = self._getcsv()
            self.unique_movies = data[["movie_id", "title"]].drop_duplicates().sort_values(by="movie_id").values.tolist()
            
        except Exception as ex:
            print('Exception:')
            print(ex)
    
    def _getcsv(self):
        container_name = os.environ.get("AZURE_CONTAINER_NAME")
        # encoding param is necessary for readall() to return str, otherwise it returns bytes
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob="ml-100k.csv")

        downloader = blob_client.download_blob(max_concurrency=1, encoding='UTF-8')
        blob_text = downloader.readall()
            
        data = pd.read_csv(StringIO(blob_text), sep=",")
        return data
    
    def get_unique_movies(self):
        return self.unique_movies
    
    