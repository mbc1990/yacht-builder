import csv
import datetime
import time

import numpy as np
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import RMSprop
from models import get_rnn, get_simple, get_simple_rnn, get_conv
import matplotlib.pyplot as plt

# Look at last 128 prices
LOOKBACK = 128

# No sampling, use every price
STEP = 1

# How many timesteps in the future we want to predict
DELAY = 1

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
            if i + bactch_size >= max_index:
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

def train():
    """
    Predicting ETH price

    Based on the temperature prediction example in the keras book
    """

    # Load and format the data
    float_data = get_float_data()
    
    # Normalize the data 
    mean = float_data[:100000].mean(axis=0)
    float_data -= mean
    std = float_data[:100000].std(axis=0)
    float_data /= std
    

    train_gen = generator(float_data,
                          lookback=LOOKBACK,
                          delay=DELAY,
                          min_index=0,
                          max_index=100000,
                          shuffle=True,
                          step=STEP,
                          batch_size=BATCH_SIZE)

    val_gen = generator(float_data,
                        lookback=LOOKBACK,
                        delay=DELAY,
                        min_index=100001,
                        max_index=200000,
                        shuffle=True,
                        step=STEP,
                        batch_size=BATCH_SIZE)

    test_gen = generator(float_data,
                         lookback=LOOKBACK,
                         delay=DELAY,
                         min_index=200001,
                         max_index=300000,
                         shuffle=True,
                         step=STEP,
                         batch_size=BATCH_SIZE)
    
    val_steps = (300000 - 200001 - LOOKBACK)
    test_steps = (len(float_data) - 300001 - LOOKBACK)

    # model = get_rnn(float_data)
    # model = get_lstm(float_data)
    # model = get_simple(float_data, lookback, step)
    model = get_conv(float_data)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=400,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps/128)
    model.save('conv-20.h5')


def get_float_data():
    txns = []
    with open('txns2.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            price = float(r[2])
            strdate = r[3]
            date = datetime.datetime.strptime(strdate, 
                                              '%Y-%m-%d %H:%M:%S')
            unix_time = time.mktime(date.timetuple())
            txns.append((unix_time, price))
    float_data = np.zeros((len(txns), 2))
    for index, line in enumerate(txns):
        values = [float(x) for x in line]
        float_data[index, :] = values
    return float_data


def chart_predictions(weights_file):
    """
    Generates a chart with real prices and predicted prices
    """
    # Load and format the data
    fload_data = get_float_data()
    
    # Normalize the data 
    mean = float_data[:100000].mean(axis=0)
    float_data -= mean
    std = float_data[:100000].std(axis=0)
    float_data /= std

    val_gen = generator(float_data,
                        lookback=LOOKBACK,
                        delay=DELAY,
                        min_index=100001,
                        max_index=200000,
                        shuffle=True,
                        step=STEP,
                        batch_size=BATCH_SIZE)

    model = load_model(weights_file)
    output = model.predict_generator(val_gen,
                                     steps=100000/128)
    # Denormalize to get predicted prices
    output *= std[1]
    output += mean[1]

    # Actual prices 
    actual = []
    x_vals = []
    real_y_vals = []
    predicted_y_vals = []
    for i in range(100001 + 128, 200000):
        val = float_data[i]

        timestamp = val[0] * std[0]
        timestamp += mean[0]

        price = val[1] * std[1]
        price += mean[1]
        output_idx = i - (100001 + 128)
        predicted_price = output[output_idx][0]
        x_vals.append(timestamp)
        real_y_vals.append(price)
        predicted_y_vals.append(predicted_price)
    
    # Plot the results
    plt.plot(x_vals, real_y_vals)
    plt.plot(x_vals, predicted_y_vals)
    plt.savefig('predictions.png')


def main():
    train()
    # chart_predictions('conv-20.h5')


if __name__ == '__main__':
    main()
