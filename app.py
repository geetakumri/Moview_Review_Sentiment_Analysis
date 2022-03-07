from cmath import phase
from flask import Flask, render_template,request, jsonify
from os.path import join
from flask_cors import CORS
from graphviz import render
import joblib
import pandas as pd
import datapreprocessing as proc
import predict
import json

app = Flask(__name__)
CORS(app)

model = joblib.load(open('nb_clf', 'rb'))

@app.route('/hello')
def hello():
    #return {'message':'successful', 'result': 'hello'}
    return render_template('index.html')

@app.route('/prediction', methods=['POST','GET'])
def prediction():
  
    '''json_string = request.data
    a_json = json.loads(json_string)
    data_df = pd.DataFrame(a_json, index=[0])   

    preprocessed_test_df = proc.perform_data_preprocessing_on_test_data(data_df)
    res = predict.perform_prediction(preprocessed_test_df)
    
    sentiment = '{sentiment}'.format(sentiment = res['Sentiment'])
    return {'message':'successful', 'result':sentiment},200'''
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"

    if request.method == 'POST':

        form = request.form.to_dict()
        print(form)

        phraseId = form['PhraseID']
        sentenceId = form['SentenceId']
        phrase = form['Phrase']

        print(f'Geeta {phraseId}, {sentenceId}, {phrase}')

        try:
            data_df = pd.DataFrame({'PhraseId':phraseId,'SentenceId':sentenceId,'Phrase':phrase},index=[0])
            print("dataframe is :", data_df)
            preprocessed_test_df = proc.perform_data_preprocessing_on_test_data(data_df)
            res = predict.perform_prediction(preprocessed_test_df)
            print("result : ",res)
        except ValueError:
            return "please enter valid values"

    return render_template('index.html', prediction_text='Movie sentiment is {}'.format(res))  

if __name__ == '__main__':
    app.run(debug = True, port = 8000)