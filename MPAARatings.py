import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import time as time
from keras.callbacks import CSVLogger
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity

tic = time.perf_counter()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

csv_logger = CSVLogger('training.log')

data = pd.read_csv("Data/IMDB_MOVIES.csv")
data2 = pd.read_csv("Data/IMDB_MOVIES_2.csv")
df1 = data[['Rated', 'Plot']]
df2 = data2[['Rated', 'Plot']]
df1 = pd.concat([df1, df2])

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
max_words = 40

dataList = tf.keras.preprocessing.sequence.pad_sequences(sequences=features, padding='post', maxlen=max_words)

dataset = tf.data.Dataset.from_tensor_slices((dataList, labels)).batch(84, drop_remainder=True).repeat(5)

dataset_shuffled = dataset.shuffle(cT)

testing = dataset_shuffled.take(test_size)
training = dataset_shuffled.skip(test_size)

train_dataset = training.shuffle(cT, reshuffle_each_iteration=True)
test_dataset = testing.shuffle(test_size)


def set_pruning_params(final_sparsity, begin_step, frequency, end_step):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0,
                                                     final_sparsity=final_sparsity,
                                                     begin_step=begin_step,
                                                     end_step=end_step,
                                                     frequency=frequency)
    }
    return pruning_params


def make_fully_connected_model(embed_dim, embed_out, output_bias=None):
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


def make_pruned_fc_model(embed_dim, embed_out, pp):
    normModel = make_fully_connected_model(embed_dim, embed_out)
    save_model("FCNN", normModel)

    pruned_model = prune_loaded_model("Models/FCNN", pp)
    return pruned_model


def make_lstm_model(embed_dim, embed_out, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words),
        tf.keras.layers.LSTM(embed_out * 4, return_sequences=True),
        tf.keras.layers.LSTM(embed_out),
        tf.keras.layers.Dense(embed_out, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax', bias_initializer=output_bias),
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def make_pruned_lstm_model(embed_dim, embed_out, pp, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.LSTM(embed_out * 4, return_sequences=True)),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.LSTM(embed_out), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(embed_out, activation='relu'), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(4, activation='sigmoid', bias_initializer=output_bias), **pp),
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def make_lstm_model2(embed_dim, embed_out):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words),
        tf.keras.layers.LSTM(embed_out * 4, return_sequences=True),
        tf.keras.layers.LSTM(embed_out * 4, go_backwards=True, return_sequences=True),
        tf.keras.layers.LSTM(embed_out),
        tf.keras.layers.Dense(embed_out, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def make_lstm_pruned_model2(embed_dim, embed_out, pp):
    model = tf.keras.Sequential([
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.LSTM(embed_out * 4, return_sequences=True), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.LSTM(embed_out * 4, go_backwards=True, return_sequences=True), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.LSTM(embed_out), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(embed_out, activation='relu'), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(4, activation='softmax'), **pp)
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


def make_simple_pruned_RNN_model(embed_dim, embed_out, pp):
    model = tf.keras.Sequential([
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Embedding(embed_dim, embed_out, input_length=max_words), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.SimpleRNN(embed_out)),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(embed_out, activation='relu'),
                                                 **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(4, activation='softmax'), **pp)
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


def make_pruned_GRU_model(embed_dim, embed_out, pp):
    model = tf.keras.Sequential([
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Embedding(embed_dim, embed_out), **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.GRU(embed_out)),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(embed_out, activation='relu'),
                                                 **pp),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(4, activation='softmax'), **pp)
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


def make_pruned_CNN_model(embed_dim, embed_out, pp):
    try:
        model = tf.keras.models.load_model('Models/CNN')
    except:
        normModel = make_CNN_model(embed_dim, embed_out)
        save_model("CNN", normModel)
        # model = train_model("CNN", normModel)

    pruned_model = prune_loaded_model("Models/CNN", pp)
    return pruned_model


def save_model(model_name, model):
    net_mod = model
    net_mod.save('Models\{}'.format(model_name))


callbacks = [
    csv_logger,
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir="logs/", profile_batch=0)
]

epochs = 150
batch_size = 20


def train_model(model, graph_title, file_name):
    net_mod = model
    #print(net_mod.summary())
    history = net_mod.fit(train_dataset, epochs=epochs, class_weight=class_weights, callbacks=callbacks, verbose=0)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.plot([0, epochs], [1, 1], 'g--', alpha=0.4)
    plt.title('Training metrics for ' + graph_title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.xlim(0, epochs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig('Graphs/' + file_name)
    plt.close()

    return net_mod


def prune_loaded_model(file_path, pp):
    model_in = tf.keras.models.load_model(file_path)

    try:
        model_in = tfmot.sparsity.keras.prune_low_magnitude(model_in, **pp)
    except ValueError:
        print("Model has unprunable layer, continuing without pruning")
    finally:
        model_in.compile(optimizer='adam',
                         loss=keras.losses.CategoricalCrossentropy(),
                         metrics=['accuracy'])
    return model_in


dense_models = []
dense_names = ["LSTM v2", "LSTM v1", "Fully Connected Model", "Simple Recurrent Neural Network",
               "Convolutional Neural Network", "Gated Recurrent Unit Model"]

running_model1 = make_lstm_model2(token.num_words, 64)
running_model2 = make_lstm_model(token.num_words, 64)
running_model3 = make_fully_connected_model(token.num_words, 8)
running_model4 = make_simpleRNN_model(token.num_words, 64)
running_model5 = make_CNN_model(token.num_words, 64)
running_model6 = make_GRU_model(token.num_words, 64)

dense_models.append(running_model1)
dense_models.append(running_model2)
dense_models.append(running_model3)
dense_models.append(running_model4)
dense_models.append(running_model5)
dense_models.append(running_model6)
counter = 0

for model in dense_models:
    print(dense_names[counter])
    train_model(model, dense_names[counter], dense_names[counter] + "/" + dense_names[counter] + "dense")
    counter += 1

sparsities = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

for y in range(0, 6):
    perf = []
    for x in range(11):
        fs = x / 10
        if fs == 1:
            fs = 0.99
        pp = set_pruning_params(fs, 0, 10, batch_size * epochs)
        file_name = dense_names[y] + "/" + str(dense_names[y] + "_" + str(fs * 100) + "sparsity").replace('.0', '')
        title_name = dense_names[y] + " " + str(fs * 100) + "% Sparsity"
        print(title_name)
        model_name = ""
        if y == 0:
            lstm2 = train_model(make_lstm_pruned_model2(token.num_words, 64, pp), title_name, file_name)
            results = lstm2.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "LSTM v2"
        if y == 1:
            lstm1 = train_model(make_pruned_lstm_model(token.num_words, 64, pp), title_name, file_name)
            results = lstm1.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "LSTM v1"
        if y == 2:
            fully_con = train_model(make_pruned_fc_model(token.num_words, 8, pp), title_name, file_name)
            results = fully_con.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "Fully Connected"
        if y == 3:
            rnn = train_model(make_simple_pruned_RNN_model(token.num_words, 64, pp), title_name, file_name)
            results = rnn.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "Simple RNN"
        if y == 4:
            cnn = train_model(make_pruned_CNN_model(token.num_words, 64, pp), title_name, file_name)
            results = cnn.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "CNN"
        if y == 5:
            gru = train_model(make_pruned_GRU_model(token.num_words, 64, pp), title_name, file_name)
            results = gru.evaluate(test_dataset, verbose=0)
            perf.append(results[1])
            model_name = "GRU"

    plt.plot(sparsities, perf)
    plt.title('Accuracies vs. Sparsity % for ' + model_name)
    plt.xlabel('Percent Sparsity(%)')
    plt.ylabel('Evaluated Accuracy')
    plt.xlim(0, 100)
    plt.xticks(sparsities)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig('Graphs/Evaluated/' + model_name)
    plt.close()
    perf.clear()

toc = time.perf_counter()

print(f"Training completed in {toc - tic:0.4f} seconds")

# try:
#     model = tf.keras.models.load_model('Checkpoints/make_model')
# except:
#     model = train_save_model("make_model", running_model)


# running_model.evaluate(test_dataset)
