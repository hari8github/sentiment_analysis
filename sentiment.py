from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
app = Flask(__name__)

model = load_model('models/sentiment_model_nlp.keras')

@app.route('/', methods = ['GET', 'POST'])
def senti():
    review = None
    sentiment = None
    if request.method == 'POST':
        review = request.form['review']
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([review])
        sequence = tokenizer.texts_to_sequences([review])
        padded_seq = pad_sequences(sequence, maxlen = 50)
        
        prediction = model.predict(padded_seq)
        if prediction > 0.6:
            sentiment = "Positive"
        elif prediction < 0.4:
            sentiment = "Negative"
        else:
            sentiment = "Ambiguous"

    return render_template('senti.html', review = review, sentiment = sentiment)

if __name__ == "__main__":
    app.run(debug = True)
