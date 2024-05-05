from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_feedforward_network(input_dim: int, width: int, depth: int, output_dim: int = 1) -> Sequential:
    """
    Creates a feedforward neural network with specified width and depth.

    :param input_dim: number of input features
    :param width: number of neurons in each hidden layer
    :param depth: number of hidden layers
    :param output_dim: number of output neurons, default is 1
    :return: a Keras Sequential model
    """
    model = Sequential()

    # Input layer
    model.add(Dense(width, input_dim=input_dim, activation='relu'))

    # Hidden layers
    for _ in range(depth):
        model.add(Dense(width, activation='relu'))

    # Output layer
    model.add(Dense(output_dim, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    return model

if __name__ == '__main__':
    """
    Debugging code here
    """
    model = create_feedforward_network(10, 32, 3)
    model.summary()