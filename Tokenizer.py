import tensorflow as tf
import pandas as pd

data = pd.read_csv("Data/IMBD_MOVIES.csv")

print(data.info())
df1 = data[['Title','Rated','Plot']]


def getDictionary(csvFile):
    dictionary = []
    for x in range(len(df1.index)):
        dictionary.append(str(df1.values[x][2]))

    return dictionary


token = tf.keras.preprocessing.text.Tokenizer(char_level=False, split=' ')
token.fit_on_texts(getDictionary(df1))
print(token.word_index)
print(token.texts_to_sequences(["his with her"]))
token.sequences_to_texts()


def textToInt(input_text, tokenizer):
    return tokenizer.text_to_sequences([input_text])


def intToText(input_sequence, tokenizer):
    return tokenizer.sequences_to_texts(input_sequence)