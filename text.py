import csv
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 


def make_embedding():
    """
    Returns a tuple of word_index, numpy array of embeddings 
    """
    texts = []
    with open('comments.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            text = r[3]
            texts.append(text)
    MAX_NUM_WORDS = 10000

    # This was 500, but that OOM's my computer :(
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_VECTOR_LENGTH = 300
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Map of string -> integer representation 
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    model = Sequential()
    model.add(Embedding(len(word_index), EMBEDDING_VECTOR_LENGTH, input_length=MAX_SEQUENCE_LENGTH))
    model.compile('rmsprop', 'mse')

    # Embedding matrix
    output_array = model.predict(data)
    return word_index, output_array

def prepare_data(word_index, word_embedding):
    """
    Reads the comments and transactions CSVs, formats the data for input
    and returns training + validation inputs formatted for the model
    """
    pass

def get_model():
    """
    Returns an LSTM based model
    """
    pass

def main():
    # Make word embeddings over *all* comments
    word_index, word_embedding = make_embedding()
    model = get_model():


if __name__ == '__main__':
    main()
