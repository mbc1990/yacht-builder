import csv
from keras.models import Sequential, load_model
from keras.layers import Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from models import get_rnn, get_simple, get_simple_rnn, get_conv
import numpy as np
import matplotlib.pyplot as plt

LOOKBACK = 128 

# No sampling, use every price
STEP = 1

# How many timesteps in the future we want to predict
DELAY = 25 

# Number of data points processed in a single batch
BATCH_SIZE = 128

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = max_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for idx, row in enumerate(rows):
            indices = range(rows[idx] - lookback, rows[idx], step)
            samples[idx] = data[indices]
            targets[idx] = data[rows[idx] + delay][1]
        yield samples, targets

class CommentPredictor(object):

    def __init__(self):
        self._load_comments()
        self.make_embedding()
        self.get_float_data()

    def _load_comments(self):
        out = []
        f = open('comments3.csv', 'rb')
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            out.append(r)
        f.close()
        self.comments = out

    def get_features_for_time(self, unix_time):
        """
        Returns a feature vector for a given unix time.
        The current algorithm is to take the average of all words in all comments
        in the lookback period
        """
        COMMENT_LOOKBACK_SECS = 60 * 5
        in_window = []
        for r in self.comments:
            date = int(r[2])
            text = r[1]
            if date >= unix_time:
                break
            if date + COMMENT_LOOKBACK_SECS >= unix_time:
                in_window.append(text)
        out_vec = np.zeros((300))
        count = 0
        for comment in in_window:
            toks = comment.lower().split(' ')
            for tok in toks:
                # TODO: word_embedding returns an array of (50, 300) for a single word, with
                # TODO: each vector the same. Probably has something to do with max seq len
                if tok in self.word_index:
                    emb = self.word_embedding[self.word_index[tok]][0]
                    out_vec += emb
                    count += 1
        out_vec /= count
        return out_vec

    def get_float_data(self):
        # Small scale for now
        LEN_INPUT = 50000
        txns = []
        with open('txns6.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for r in reader:
                price = float(r[1])
                unix_time = int(r[2])
                txns.append((unix_time, price))

        float_data = np.zeros((LEN_INPUT, 302))
        for index, line in enumerate(txns):
            if index == LEN_INPUT:
                self.float_data = float_data
                return

            if index % 100 == 0:
                print str(index) + "/" + str(LEN_INPUT)
            values = [float(x) for x in line]
            values = np.append(values, self.get_features_for_time(line[0]))
            float_data[index, :] = values
        self.float_data = float_data

    def make_embedding(self):
        """
        Returns a tuple of word_index, numpy array of embeddings 
        """
        texts = []
        with open('comments3.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for r in reader:
                text = r[1]
                texts.append(text)

        MAX_SEQUENCE_LENGTH = 50
        EMBEDDING_VECTOR_LENGTH = 300

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Map of string -> integer representation 
        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, EMBEDDING_VECTOR_LENGTH, input_length=MAX_SEQUENCE_LENGTH))
        model.compile('rmsprop', 'mse')

        # Embedding matrix
        output_array = model.predict(data)
        self.word_index = word_index
        self.word_embedding = output_array

    def train(self ):
        # Make word embeddings over *all* comments
        print "Embedding computed"
        float_data = self.float_data

        mean = float_data[:1000].mean(axis=0)
        float_data -= mean
        std = float_data[:1000].std(axis=0)
        float_data /= std

        train_gen = generator(float_data,
                              lookback=LOOKBACK,
                              delay=DELAY,
                              min_index=0,
                              max_index=1000,
                              step=STEP,
                              batch_size=BATCH_SIZE)

        val_gen = generator(float_data,
                            lookback=LOOKBACK,
                            delay=DELAY,
                            min_index=1001,
                            max_index=2000,
                            step=STEP,
                            batch_size=BATCH_SIZE)

        model = get_conv(float_data)
        val_steps = 5000
        history = model.fit_generator(train_gen,
                                      steps_per_epoch=25,
                                      epochs=25,
                                      validation_data=val_gen,
                                      validation_steps=val_steps/128)
        model.save('text-2.h5')
        self.model = model

    def chart_predictions(self):

        float_data = self.float_data
        
        # Normalize the data 
        mean = float_data[:2000].mean(axis=0)
        float_data -= mean
        std = float_data[:2000].std(axis=0)
        float_data /= std
        test_gen = generator(float_data,
                             lookback=LOOKBACK,
                             delay=DELAY,
                             min_index=2001,
                             max_index=4000,
                             step=STEP,
                             batch_size=BATCH_SIZE)
        

        model = self.model
        output = model.predict_generator(test_gen,
                                         steps=100)
        # Denormalize to get predicted prices
        output *= std[1]
        output += mean[1]

        # Actual prices 
        actual = []
        x_vals = []
        real_y_vals = []
        predicted_y_vals = []
        for i in range(2001 + 128, 4000):
            val = float_data[i]

            timestamp = val[0] * std[0]
            timestamp += mean[0]

            price = val[1] * std[1]
            price += mean[1]

            output_idx = i - (2001 + 128)
            predicted_price = output[output_idx][0]
            x_vals.append(timestamp)
            real_y_vals.append(price)
            predicted_y_vals.append(predicted_price)
        
        # Plot the results
        plt.plot(x_vals, real_y_vals)
        plt.plot(x_vals, predicted_y_vals)
        plt.savefig('predictions.png')


def main():
    """
    Attempt #2 at text based predictions, this time building from the keras book
    """
    predictor = CommentPredictor()
    predictor.train()
    predictor.chart_predictions()

if __name__ == '__main__':
    main()
