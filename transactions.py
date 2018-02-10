import csv
import datetime
import time

import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from models import get_rnn

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

def main():
    """
    Predicting ETH price

    Based on the temperature prediction example in the keras book
    """

    # Load and format the data
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
    
    # Normalize the data 
    # TODO: With only one input (price) do we need to normalize it?
    mean = float_data[:100000].mean(axis=0)
    float_data -= mean
    std = float_data[:100000].std(axis=0)
    float_data /= std
    
    # Look at last 128 prices
    lookback = 128

    # No sampling, use every price
    step = 1
    
    # Try to predict the price 128 timestamps in the future
    # delay = 128 
    delay = 1
    
    batch_size = 128

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=100000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)

    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=100001,
                        max_index=200000,
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)

    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=200001,
                         max_index=300000,
                         shuffle=True,
                         step=step,
                         batch_size=batch_size)
    
    val_steps = (300000 - 200001 - lookback)
    test_steps = (len(float_data) - 300001 - lookback)

    model = get_rnn(float_data)
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=400,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps/128)


if __name__ == '__main__':
    main()
