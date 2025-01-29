from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
import pickle

# Initialize MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/rakesh-kumar-22/emotion-detection-mlops.mlflow")
import dagshub
dagshub.init(repo_owner='rakesh-kumar-22', repo_name='emotion-detection-mlops', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


app=Flask(__name__)
#load model from model registry
model_name='my_model'
model_version=1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))
@app.route('/')

def home ():
    return render_template("index.html",result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    
    #clean
    text=normalize_text(text)
    #bow
    features=vectorizer.transform([text])
    #predict
    result=model.predict(features)
    #return result 
    return render_template('index.html',result=result[0])
    
    return text
app.run(debug=True)