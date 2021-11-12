import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

# reading all the datasets from csv files
df = pd.read_csv('./datasets/dataset.csv')
df1 = pd.read_csv('./datasets/Symptom-severity.csv')
description = pd.read_csv('./datasets/symptom_Description.csv')
precaution = pd.read_csv('./datasets/symptom_precaution.csv')

# pre-processing datas
df.isna().sum()
df.isnull().sum()

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)

df = df.fillna(0)
df.head()

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
df = d.replace('foul_smell_of urine',0)
df.head()

(df[cols] == 0).all()

df['Disease'].value_counts()

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values


# loading the trained (pickled) model
model = pickle.load(open('./model/model.sav', 'rb'))

# running flask app
app = Flask(__name__)
CORS(app)


# function to predict disease based on symptoms
def SVM(symptoms, loc):
    
    sym = np.array(df1["Symptom"])
    wei = np.array(df1["weight"])

    for j in range(len(symptoms)):
        for k in range(len(sym)):
            if symptoms[j]==sym[k]:
                symptoms[j]=wei[k]

    total_length = 17
    zeros_required = total_length - len(symptoms)
    
    nulls_required = [0] * zeros_required

    total_symptoms = [symptoms + nulls_required]

    pred2 = model.predict(total_symptoms)

    return list(pred2)


@app.route('/', methods = ['POST'])
def index():

    # read the request body for symptoms and location
    symptoms = request.json['symptoms']
    location = request.json['location']

    # predict the disease
    res = SVM(symptoms, location)
    res = res[0]

    # get the description and precautions for the predicted disease
    des = description[description.Disease == res]['Description'].item()
    prec = precaution[precaution.Disease == res]['Precaution_1'].item()
    des = str(des)
    prec = str(prec)
    
    # return the result
    return jsonify(
        disease=res,
        description=des,
        precaution=prec
    ), 200

@app.route('/disease')
def disease():

    # all the possible symptoms
    dis =  ["fatigue", "yellowish_skin", "loss_of_appetite", "yellowing_of_eyes", 'family_history',"stomach_pain", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]
    return jsonify(
        response=dis
    ), 200

@app.route('/location')
def location():

    # all the possible locations
    loc =  ["New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    return jsonify(
        response=loc
    ), 200


if __name__ == '__main__':
    app.debug = True
    app.run()