from flask import Flask,render_template,url_for,request
from textblob import TextBlob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


app=Flask(__name__)
Swagger(app)

model = pickle.load(open('nb.pkl', 'rb'))
model2 = pickle.load(open('nbchi.pkl','rb'))
countVect = pickle.load(open('vectorizer.pkl','rb'))
countVect2 = pickle.load(open('vectorizerchi.pkl','rb'))


@app.route('/')
def main():
    return render_template('home.html')

@app.route("/sentiment_analyzer")
def sentiment_analyzer():
    return render_template('sentiment_analyzer.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sentiment_logic', methods=['POST'])
def sentiment_logic():
    print(request.form['algoritma'])

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        vect2 = countVect2.transform(data).toarray()
        algoritma=request.form['algoritma']
        if algoritma == 'nb':
            my_prediction = model.predict(vect)
        else:
            my_prediction = model2.predict(vect2)

    return render_template('sentiment_analyzer.html', Reviews=Reviews, prediction = my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)



