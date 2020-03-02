from flask import Flask, request, json
import boto3
import pickle
import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection


BUCKET_NAME = 'newsstorage'
MODEL_FILE_NAME = 'lr_model.pkl'
MODEL_LOCAL_PATH = './' + MODEL_FILE_NAME
app = Flask(__name__)
S3 = boto3.client('s3', region_name='ap-southeast-1')
@app.route('/', methods=['POST'])
def index():    
    # Parse request body for model input 
    body_dict = json.loads(request.get_data().decode('utf-8')) 
    data = body_dict['0']     
    
    # Load model
    model = load_model(MODEL_FILE_NAME)
# Make prediction 
    prediction = model.predict(data).tolist()
# Respond with prediction result
    result = {'prediction': prediction}    
   
    return json.dumps(result)


def load_model(key):
     # Load model from S3 bucket
    #key_obj = Key(BUCKET_NAME)
    #key_obj.key = MODEL_FILE_NAME
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
# Load pickle model
    model_str = response['Body'].read()     
    model = pickle.loads(model_str)
   
    return model

'''def predict(data):
      # Process your data, create a dataframe/vector and make your predictions
    final_formatted_data = []
    return load_model().predict(final_formatted_data)'''