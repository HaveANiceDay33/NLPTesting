import tensorflow as tf
import pandas as pd

data = pd.read_csv("Data/IMDB_MOVIES.csv")
df1 = data[['Title', 'Rated', 'Plot']]


def getDictionary(csvFile):
    dictionary = []
    for x in range(len(df1.index)):
        dictionary.append(str(df1.values[x][2]))
    return dictionary


token = tf.keras.preprocessing.text.Tokenizer(num_words=16000, char_level=False, split=' ')
token.fit_on_texts(getDictionary(df1))
print(token.word_index)


def textToInt(input_text, tokenizer):
    return tokenizer.text_to_sequences([input_text])


def intToText(input_sequence, tokenizer):
    return tokenizer.sequences_to_texts(input_sequence)


vocab_dim = token.num_words


def makeModel(embed_dim, embed_out, lst_dim, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=embed_dim, output_dim=embed_out,batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.LSTM(lst_dim, return_sequences=True))
    model.add(tf.keras.layers.LSTM(lst_dim))
    model.add(tf.keras.layers.Softmax())
    return model


model = makeModel(token.num_words, 1000, 256, 256)
print(model.summary())
