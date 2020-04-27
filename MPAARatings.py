from datetime import time

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

data = pd.read_csv("Data/IMDB_MOVIES.csv")
data2 = pd.read_csv("Data/IMDB_MOVIES_2.csv")
df1 = data[['Rated', 'Plot']]
df2 = data2[['Rated', 'Plot']]
df1 = pd.concat([df1, df2])

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# data filtering
df1 = df1.replace(to_replace='NOT RATED', value=np.nan)
df1 = df1.replace(to_replace='UNRATED', value=np.nan)
df1 = df1.replace(to_replace='APPROVED', value=np.nan)
df1 = df1.replace(to_replace='PASSED', value=np.nan)
df1 = df1.replace(to_replace='GP', value=np.nan)
df1 = df1.replace(to_replace='TV-14', value=np.nan)
df1 = df1.replace(to_replace='TV-Y7', value=np.nan)
df1 = df1.replace(to_replace='TV-Y', value=np.nan)
df1 = df1.replace(to_replace='M/PG', value=np.nan)
df1 = df1.replace(to_replace='TV-G', value=np.nan)
df1 = df1.replace(to_replace='TV-PG', value=np.nan)
df1 = df1.replace(to_replace='TV-MA', value=np.nan)
df1 = df1.replace(to_replace='NC-17', value=np.nan)
df1 = df1.replace(to_replace='M', value=np.nan)
df1 = df1.replace(to_replace='X', value=np.nan)
df1 = df1.dropna(axis=0)
df1 = df1.reset_index()
df1.pop('index')

# split the dataframe
target = df1.pop('Rated')

# creates a list of the plots
temp = df1.to_numpy().tolist()

text = []
for i in range(len(temp)):
    text.append(temp[i][0])

# create the tokenizer and fit it on the plots
token = tf.keras.preprocessing.text.Tokenizer(num_words=13561, char_level=False, split=' ',
                                              filters='!"#$%&-()*+,./:;<=>?@[\\]^_`{|}~\t\n')
token.fit_on_texts(text)


def text_to_int(input_text, tokenizer):
    return tokenizer.texts_to_sequences([input_text])[0]


def int_to_text(input_sequence, tokenizer):
    return tokenizer.sequences_to_texts(input_sequence)


features = []
labels = []

cR = 0
c13 = 0
cPG = 0
cG = 0
cT = 0

for i in range(len(df1.values)):
    features.append(text_to_int(df1.values[i][0], token))

    if target.values[i] == 'R':
        labels.append([1, 0, 0, 0])
        cR += 1
    elif target.values[i] == 'PG-13':
        labels.append([0, 1, 0, 0])
        c13 += 1
    elif target.values[i] == 'PG':
        labels.append([0, 0, 1, 0])
        cPG += 1
    elif target.values[i] == 'G':
        labels.append([0, 0, 0, 1])
        cG += 1
    else:
        labels.append(5)
    cT += 1

weight_for_r = (1 / cR) * cT / 10
weight_for_pg13 = (1 / c13) * cT / 10
weight_for_pg = (1 / cPG) * cT / 10
weight_for_g = (1 / cG) * cT / 10

class_weights = {0: weight_for_r, 1: weight_for_pg13, 2: weight_for_pg, 3: weight_for_g}

test_size = 500
max_words = 25

dataList = tf.keras.preprocessing.sequence.pad_sequences(sequences=features, padding='post', maxlen=max_words)

dataset = tf.data.Dataset.from_tensor_slices((dataList, labels)).batch(8)
dataset_shuffled = dataset.shuffle(cT)

testing = dataset_shuffled.take(test_size)
training = dataset_shuffled.skip(test_size)

train_dataset = training.shuffle(cT)
test_dataset = testing.shuffle(test_size)

def make_model(embed_dim, embed_out, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(embed_dim, embed_out, input_length=max_words))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(max_words * embed_out, activation='relu'))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(4, activation='sigmoid', bias_initializer=output_bias))
    model.add(keras.layers.Softmax())

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def make_lstm_model(embed_dim, embed_out, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(embed_dim, embed_out, input_length=max_words))
    model.add(keras.layers.SpatialDropout1D(0.1))
    # model.add(keras.layers.LSTM(max_words, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(max_words))
    model.add(keras.layers.Dense(max_words * embed_out, activation='relu'))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(4, activation='sigmoid', bias_initializer=output_bias))
    model.add(keras.layers.Softmax())

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def make_lstm_model2(embed_dim, embed_out):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_out)),
        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embed_out, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def make_simpleRNN_model(embed_dim, embed_out):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words),
        tf.keras.layers.SimpleRNN(embed_out),
        tf.keras.layers.Dense(embed_out, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def make_GRU_model(embed_dim, embed_out):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words),
        tf.keras.layers.GRU(embed_out),
        tf.keras.layers.Dense(embed_out, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def make_CNN_model(embed_dim, embed_out):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out),
        tf.keras.layers.Conv1D(kernel_size=max_words, filters=5, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model



def train_save_model(model_name, model):
    net_mod = model
    print(net_mod.summary())
    tb = TensorBoard(log_dir='logs\{}'.format(model_name))
    net_mod.fit(train_dataset, epochs=5, class_weight=class_weights, callbacks=[tb])
    net_mod.save('Checkpoints\{}'.format(model_name))

    return net_mod

#running_model = make_lstm_model2(token.num_words, 64)
#running_model = make_lstm_model(token.num_words, 64)
#running_model = make_fully_connected_model(token.num_words, 8)
#running_model = make_simpleRNN_model(token.num_words, 64)
#running_model = make_CNN_model(token.num_words, 64)
running_model = make_GRU_model(token.num_words,64)


try:
    model = tf.keras.models.load_model('Checkpoints/Test_Model')
except:
    model = train_save_model("Test_Model", running_model)

model.evaluate(test_dataset)