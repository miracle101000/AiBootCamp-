import numpy as np
 
 # Sigmoid activation functions adn its derivatives
def sigmoid(x):
     return 1/1(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x) 

# Mean Squared Error (MSE) Loss Function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Basic Neural Network Class
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weight_hidden_output =  np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_ouput = np.random.randn(1, output_size)
    
    # Forward Pass
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weight_hidden_output) + self.bias_ouput
        self.output = sigmoid(self.output_input)
        return self.output
    
    # Backward pass and weights update
    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.weight_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_ouput += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward Pass
            output = self.forward(X)
            
            self.backward(X, y, output, learning_rate)
            
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")

# XOR dataset
X = np.array([[0,0], [0,1], [1,1]])     
y = np.array([[0], [1], [1], [0]])     

nn =  BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

nn.train(X, y, epochs=10000,learning_rate=.1)     

print("\nTest the trained neural network:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.forward(X[i])}, Actual Output: {y[i]}")
            

        
        