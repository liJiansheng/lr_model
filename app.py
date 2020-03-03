import pandas as pd
from flask import Flask, jsonify, request,json
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # get data
   
    body_dict = json.loads(request.get_data().decode('utf-8')) 
    data = body_dict['0']
    # predictions

    prediction=[]
    for v in data.values():
        p=model.predict([v]).tolist()
        #print(p)
        prediction.append(p[0])
#prediction = model.predict([data['0']]).tolist()
        #print(prediction)
    result = {'prediction': prediction}   

    # return data
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)