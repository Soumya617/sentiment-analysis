# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords
#from pattern.en import lemma
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
            text = request.form['Review']
            a=set(stopwords.words('english'))
            a.remove("not")
            review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
            review = review.lower()
            review_words = review.split()
            review_words = [word for word in review_words if not word in a]
            lem=WordNetLemmatizer()
            review = [lem.lemmatize(word) for word in review_words]
            review = ' '.join(review)
            data=[review]
       
            vectorizer = cv.transform(data).toarray()
            prediction = model.predict(vectorizer)
    if prediction:
        return render_template('index.html', prediction_text='The review is Postive')
    else:
        return render_template('index.html', prediction_text='The review is Negative.')


if __name__ == "__main__":
    app.run(debug=True)