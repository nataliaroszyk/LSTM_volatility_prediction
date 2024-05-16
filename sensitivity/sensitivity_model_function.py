from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam

def create_model_sensitivity(input_shape, loss_function='mean_squared_error', learning_rate=0.001, 
                 lstm_layers=2, activation_function='tanh'):
    model = Sequential()

    # Adding LSTM layers based on the specified number of layers
    for i in range(lstm_layers):
        if i == 0:
            # First layer needs to specify input shape
            model.add(LSTM(128, return_sequences=True if lstm_layers > 1 else False, 
                           activation=activation_function, input_shape=input_shape))
        else:
            # Subsequent layers only return sequences if they're not the last or only one layer
            model.add(LSTM(128, return_sequences=False if i == lstm_layers - 1 else True, 
                           activation=activation_function))
        model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(1, activation='relu'))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function)
    return model
