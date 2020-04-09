import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("Data/IMBD_MOVIES.csv")
df1 = data[['Rated', 'Plot']]

df1 = df1.replace(to_replace='NOT RATED', value=np.nan)
df1 = df1.replace(to_replace='UNRATED', value=np.nan)
df1 = df1.replace(to_replace='APPROVED', value=np.nan)
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
text.extend(['PG', 'R', 'PG-13', 'NC-17', 'G'])

# print(text)
# create the tokenizer and fit it on the plots
token = tf.keras.preprocessing.text.Tokenizer(num_words=16000, char_level=False, split=' ',
                                              filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
token.fit_on_texts(text)


def text_to_int(input_text, tokenizer):
    return tokenizer.texts_to_sequences([input_text])[0]


def int_to_text(input_sequence, tokenizer):
    return tokenizer.sequences_to_texts(input_sequence)


intList = []
targetList = []
for i in range(len(df1.values)):
    intList.append(text_to_int(df1.values[i][0], token))
    targetList.append(text_to_int(target.values[i][0], token))

dataList = tf.keras.preprocessing.sequence.pad_sequences(sequences=intList, padding='post', maxlen=25)
dataset = tf.data.Dataset.from_tensor_slices((dataList, targetList))

train_dataset = dataset.shuffle(len(targetList)).batch(1)

'''
#Model stuff doesn't really work at all

def makeModel(embed_dim, embed_out, lst_dim, batch_size):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(input_length=25, output_dim=embed_out,input_dim=embed_dim))
    model.add(tf.keras.layers.LSTM(lst_dim, return_sequences=True))
    model.add(tf.keras.layers.LSTM(lst_dim))
    model.add(tf.keras.layers.Dense(5, activation='relu', ))
    model.add(tf.keras.layers.Softmax())
    return model


netMod = makeModel(token.num_words,256, 256, 256)
netMod.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print(netMod.summary())

netMod.fit(train_dataset,epochs=3)

'''