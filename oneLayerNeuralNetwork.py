'''
    ///// One-layer Neural Network ////
    
    Credits to Siraj Raval
    Modified by Sebastian Rivera
'''
from numpy import random, exp, dot, array


class OneLayerNeuralNetwok():
    def __init__(self):
        # Seed the random funtion ensures that it generates the same numbers
        # on each iteration
        random.seed(1)

        # Model of a single neuron, with 5 input connections and 1 output.
        # The weights are stores on a 5 x 1 matrix 
        # with values are in range [-1,1]
        self.synaptic_weights = 2 * random.random((5,1)) - 1

    # Sigmoid function that will be used as the activation function
    # https://en.wikipedia.org/wiki/Sigmoid_function
    # In: Weighted sum of the inputs 
    # Out: Normalization between 0 and 1
    # When asked for the derivative -> returns gradient of the Sigmoid curve
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + exp(-x))

    # Train the neural network through a process of trail and error. The weights 
    # will be updated on each iteration
    def train(self, training_set_inputs, training_set_outputs, training_iterations):
        for iteration in xrange(training_iterations):
            # Input the training set through the NN
            output = self.predict(training_set_inputs)

            # Calculate error = desired output - predicted output
            error = training_set_outputs - output

            # Multiply the error by the input and by the Sigmoid curve gradient
            # Less confident weights are adjusted more
            adjustment = dot(training_set_inputs.T, error * self.sigmoid(output,True))

            # Adjust weights
            self.synaptic_weights += adjustment

    # Predict with the currect weights
    def predict(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
    # Initialise a one-layer neural network
    neural_network = OneLayerNeuralNetwok()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # Training set with 5 elements
    training_set_inputs = array([
        [1,0,0,1,0],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,1,0,0,1]
    ])
    training_set_outputs = array([[0,1,0,1,0]]).T

    # Train the neural network using the training set. It will be done 10K times
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    #Test the neural network after training
    print "Considering new situation [1,1,0,0,0] -> ?: "
    print neural_network.predict(array([1,1,0,0,0]))