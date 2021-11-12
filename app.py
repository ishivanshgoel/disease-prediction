import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

df = pd.read_csv('./datasets/dataset.csv')
df1 = pd.read_csv('./datasets/Symptom-severity.csv')
description = pd.read_csv('./datasets/symptom_Description.csv')
precaution = pd.read_csv('./datasets/symptom_precaution.csv')

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

# x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.80)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# model = SVC()
# model.fit(x_train, y_train)

# pickle.dump(model, open('model.sav', 'wb'))

model = pickle.load(open('./model/model.sav', 'rb'))

# preds = model.predict(x_test)
# print('Preds', preds)

app = Flask(__name__)
CORS(app)

# print('Precaution ', precaution)


def SVM(psymptoms, loc):
    
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])

    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
    psy = [psymptoms + nulls]

    pred2 = model.predict(psy)

    return list(pred2)


@app.route('/', methods = ['POST'])
def index():
    symptoms = request.json['symptoms']
    location = request.json['location']


    res = SVM(symptoms, location)
    res = res[0]

    des = description[description.Disease == res]['Description']
    prec = precaution[precaution.Disease == res]['Precaution_1']
    des = str(des)
    prec = str(prec)
    
    return jsonify(
        disease=res,
        description=des,
        precaution=prec
    ), 200

@app.route('/disease')
def disease():
    dis =  ["fatigue", "yellowish_skin", "loss_of_appetite", "yellowing_of_eyes", 'family_history',"stomach_pain", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]
    return jsonify(
        response=dis
    ), 200

@app.route('/location')
def location():
    loc =  ["New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    return jsonify(
        response=loc
    ), 200


if __name__ == '__main__':
    app.debug = True
    app.run()