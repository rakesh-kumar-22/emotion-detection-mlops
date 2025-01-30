# load test + signature test + performance test

import unittest
import mlflow
import os
import pickle
import pandas as pd

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "rakesh-kumar-22"
        repo_name = "emotion-detection-mlops"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.new_model_uri)
        #vectorizer 
        cls.vectorizer=pickle.load(open('models/vectorizer.pkl','rb'))
        @staticmethod
        def get_latest_model_version(model_name):
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_model_version(model_name,stages=['staging'])
            return latest_version[0].version if latest_version else None
        def test_model_loaded_properly(self):
            self.assertIsNotNone(self.model)
            
        def test_model_signature(self):
            #create a dummy input for the model based on input shape
            input_text= "hi how are you"
            input_data= self.vetorize.transform([input_text])
            input_df=pd.dataframe(input_data.to_array(),columns=[str(i) for i in range(input_data.shape[1])])
            #predict using the model to check input and output shape
            prediction=self.model.predict(input_df)
            
            #verify the input shape
            self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
            self.assertEqual(len(prediction),input_df.shape[0])
            #verify the output shape
            self.assertEqual(len(prediction.shape),1)
            
if __name__ == '__main__':
    unittest.main()
