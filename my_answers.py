import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    #TK: Adding iterations
    for i in range(0, len(series)-window_size):
        sub = []
        for j in range(i, i+window_size):
            sub.append(series[j])
        X.append(sub)
        y.append(series[window_size+i])
           
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # TK: building two hidden layer RNN
    # One LSTM layer with 5 units
    # One fully connected layer with 1 unit
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1))
    return model          
    #pass


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    # For inclusion of the whole ascii
    # text = "".join([x if ord(x) < 128 else ' ' for x in s])
    text = "".join([x if (x in alphabet or x in punctuation) else ' ' for x in text])
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # TK: the range for i now depends on text length, window_size and step_size
    for i in range(0, int((len(text)-window_size)/(step_size))):
        # appending the inputs with string rather than chars of the string
        inputs.append(text[(i*step_size):(i*step_size)+window_size])
        outputs.append(text[(i*step_size)+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200,input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
    #pass
