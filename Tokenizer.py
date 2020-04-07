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