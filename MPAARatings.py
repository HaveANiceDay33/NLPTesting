import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("Data/IMBD_MOVIES.csv")
df1 = data[['Rated', 'Plot']]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#data filtering
df1 = df1.replace(to_replace='NOT RATED', value=np.nan)
df1 = df1.replace(to_replace='UNRATED', value=np.nan)
df1 = df1.replace(to_replace='APPROVED', value=np.nan)
df1 = df1.replace(to_replace='PASSED', value=np.nan)
df1 = df1.replace(to_replace='GP', value=np.nan)
df1 = df1.replace(to_replace='TV-14', value=np.nan)
df1 = df1.replace(to_replace='M/PG', value=np.nan)
df1 = df1.replace(to_replace='TV-G', value=np.nan)
df1 = df1.replace(to_replace='TV-PG', value=np.nan)
df1 = df1.replace(to_replace='TV-MA', value=np.nan)
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

# print(text)
# create the tokenizer and fit it on the plots
token = tf.keras.preprocessing.text.Tokenizer(num_words = 13561,char_level=False, split=' ',
                                              filters='!"#$%&-()*+,./:;<=>?@[\\]^_`{|}~\t\n')
token.fit_on_texts(text)


def text_to_int(input_text, tokenizer):
    return tokenizer.texts_to_sequences([input_text])[0]


def int_to_text(input_sequence, tokenizer):
    return tokenizer.sequences_to_texts(input_sequence)


intList = []
targetList = []

for i in range(len(df1.values)):
    intList.append(text_to_int(df1.values[i][0], token))
    if target.values[i] == 'NC-17':
        targetList.append([1,0,0,0,0])
    elif target.values[i] == 'R':
        targetList.append([0,1,0,0,0])
    elif target.values[i] == 'PG-13':
        targetList.append([0,0,1,0,0])
    elif target.values[i] == 'PG':
        targetList.append([0,0,0,1,0])
    elif target.values[i] == 'G':
        targetList.append([0,0,0,0,1])
    else:
        targetList.append(5)

test_size = 200

dataList = tf.keras.preprocessing.sequence.pad_sequences(sequences=intList, padding='post', maxlen=25)

dataset = tf.data.Dataset.from_tensor_slices((dataList, targetList))
dataset_shuffled = dataset.shuffle(len(targetList)-1)

testing = dataset_shuffled.take(test_size)
training = dataset_shuffled.skip(test_size)

train_dataset = training.shuffle(len(targetList)-test_size).batch(5)
test_dataset = testing.shuffle(test_size).batch(1)



#Model stuff doesn't really work at all

def makeModel(embed_dim, embed_out, lst_dim, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(embed_dim, embed_out, input_length=25))
    # model.add(tf.keras.layers.LSTM(lst_dim))
    # model.add(tf.keras.layers.LSTM(lst_dim))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lst_dim*2, return_sequences=True))),
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lst_dim)))
    model.add(tf.keras.layers.Dense(lst_dim, activation='relu')),
    model.add(tf.keras.layers.Dense(256)),
    model.add(tf.keras.layers.Dense(64)),
    model.add(tf.keras.layers.Dense(5,activation = 'relu')),
    model.add(tf.keras.layers.Softmax())
    return model

netMod = makeModel(token.num_words, 8, 256, 256)
netMod.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(netMod.summary())

netMod.fit(train_dataset, epochs=5)

netMod.evaluate(test_dataset)

