from flask import Flask, request, render_template
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
import contractions
import pickle

app = Flask(__name__)

punct = list(string.punctuation)
punct += ['rt', 'via', '...']


def clean_tweet(text, stop=False):
  """helper function for cleaning and processing tweets"""

  text = re.sub(r'&(.*?);', "", text)
  tokens = TweetTokenizer().tokenize(text=text.lower())
  tokens = [tok for tok in tokens if tok not in punct and not tok.isdigit()
                    and not tok.startswith("@") and not tok.startswith("#") and not tok.startswith("http")]

  if "you's" in tokens:
    tokens = list(map(lambda x: x.replace("you's", 'you are'), tokens))

  text = contractions.fix(' '.join(tokens))

  # Remove stopwords if true
  if stop:
    tokens = [tok for tok in text.split() if tok not in stopword_list]
    text = ' '.join(tokens)

    return text

  return text


json_file = open("hate_model.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("hate_weights.h5")

model.compile(loss="categorical_crossentropy", optimizer="adam", 
                metrics = "accuracy")

# loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/", methods=["POST"])
def predict():
    global model

    

    # seq = tokenizer.texts_to_sequences([clean_tweet(tweet)])
    # padded = pad_sequences(seq, maxlen=30)
    # pred = model.predict(padded)

    # labels = ['hate_speech', 'offensive language', 'neither']
    # response = labels[np.argmax(pred)]
    # return str(response)

    if request.method == 'POST':
        tweet = request.form['message']
        seq = tokenizer.texts_to_sequences([clean_tweet(tweet)])
        padded = pad_sequences(seq, maxlen=30)
        pred = model.predict(padded)
        
        print(pred)

        labels = ['hate_speech', 'offensive language', 'neither']
        response = labels[np.argmax(pred[0])]
        
        print(response)
        return render_template('index.html', prediction=response)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 80)
    app.run(debug=True)