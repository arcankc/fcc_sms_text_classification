import urllib.request
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


url_train = 'https://cdn.freecodecamp.org/project-data/sms/train-data.tsv'
url_test = 'https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv'


headers = {'User-Agent': 'Mozilla/5.0'}


req_train = urllib.request.Request(url_train, headers=headers)
with urllib.request.urlopen(req_train) as response, open('train-data.tsv', 'wb') as out_file:
    out_file.write(response.read())


req_test = urllib.request.Request(url_test, headers=headers)
with urllib.request.urlopen(req_test) as response, open('valid-data.tsv', 'wb') as out_file:
    out_file.write(response.read())

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"


train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])


train_df['label_num'] = train_df['label'].map({'ham': 0, 'spam': 1})
test_df['label_num'] = test_df['label'].map({'ham': 0, 'spam': 1})


from tensorflow.keras.layers import TextVectorization

max_features = 10000
sequence_length = 100

vectorizer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
vectorizer.adapt(train_df['message'].values)

train_ds = tf.data.Dataset.from_tensor_slices((
    train_df['message'].values,
    train_df['label_num'].values
))

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df['message'].values,
    test_df['label_num'].values
))

batch_size = 32
train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(max_features + 1, 32),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(train_ds, validation_data=test_ds, epochs=50)


def predict_message(pred_text):
    pred_input = tf.convert_to_tensor([pred_text])
    prediction = model.predict(pred_input)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    return [float(prediction), label]


pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)


def test_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying!")


test_predictions()
