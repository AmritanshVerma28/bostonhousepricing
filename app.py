## sample
sample = """{
"data":
{
"1":1,
"2":2,
"3":3,
"4":4,
"5":5,
"6":6,
"7":7,
"8":8,
"9":9,
"10":10,
"11":11,
"12":12,
"13":13
}

}"""

import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import sys

print(sys.version)

import numpy as np


app = Flask(__name__)


## Load model

regmodel =  pickle.load(open("regmodel.pkl","rb"))
scaler =  pickle.load(open("scaling.pkl","rb"))

###
@app.route("/")
def home():
    home_ = render_template("home.html")
    return home_

@app.route("/predict_api",methods=['POST'])
def predict_api():
    print("predict_api called")

    data = request.json['data']
    print("here 1")
    print(data)
    data_ = np.array(list(data.values())).reshape(1,-1)
    print(data_)
    new_data = scaler.transform(data_)
    op = regmodel.predict(new_data)
    op_j =  jsonify(op[0])
    print(op_j)
    print("predict_api called end")
    return op_j

@app.route("/predict",methods=['POST'])
def predict():
    print("t1")
    data = [float(x) for x in request.form.values()]
    print("t2")
    data_ = np.array(data).reshape(1,-1)
    print(data_)
    print("t3")
    new_data = scaler.transform(data_)
    op = regmodel.predict(new_data)[0]
  #  op_j =  jsonify(op[0])
   # print(op_j)
    print("predict called end")
    home_new = render_template("home.html",prediction_text="Price is "+str(op))



    return home_new

if __name__ =="__main__":
    app.run(debug=True)

#kill -9 $(lsof -t -i:5000)
#https://stackoverflow.com/questions/11583562/how-to-kill-a-process-running-on-particular-port-in-linux