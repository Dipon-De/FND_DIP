from flask import Flask , request , render_template
import pickle
import sklearn
import numpy

vector = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("news_det_model.pkl",'rb'))

import os
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news = request.form['news']
        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        return render_template("prediction.html", prediction_text=f"News Headline is  -->  {predict}")
    else:
        return render_template("prediction.html")



if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)