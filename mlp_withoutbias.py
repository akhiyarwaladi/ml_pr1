from numpy import exp, array, random, dot, zeros
import math

class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons))
		#self.synaptic_weights = zeros((number_of_inputs_per_neuron, number_of_neurons))


class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2
		self.learning_rate = 0.2

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x, response=1):
		#return 1 / (1 + math.e**(-(x)/response))
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			
			print("================================================================")
			print("               EPOCH" + str(iteration) + "               ")
			print("================================================================")
			# Pass the training set through our neural network
			output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)
			print("output layer 1\n" + str(output_from_layer_1) + "\n"+ str(output_from_layer_1.shape))
			print("output layer 2\n" + str(output_from_layer_2) + "\n"+ str(output_from_layer_2.shape))

			# Calculate the error for layer 2 (The difference between the desired output
			# and the predicted output).
			layer2_error = training_set_outputs - output_from_layer_2
			print("layer2 error")
			print(layer2_error.T)

			#print(layer2_error)
			if(layer2_error.all() == 0.0):
				break

			layer2_delta = self.learning_rate * layer2_error * self.__sigmoid_derivative(output_from_layer_2)
			print("layer2 delta")
			print(layer2_delta.T)

			# Calculate the error for layer 1 (By looking at the weights in layer 1,
			# we can determine by how much layer 1 contributed to the error in layer 2).
			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			print("layer1 error")
			print(layer1_error.T)
			layer1_delta = self.learning_rate * layer1_error * self.__sigmoid_derivative(output_from_layer_1)
	

			# Calculate how much to adjust the weights by
			layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
			layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
			print("layer2 adjustments")
			print(layer2_adjustment)

			# Adjust the weights.
			self.layer1.synaptic_weights += layer1_adjustment
		
			self.layer2.synaptic_weights += layer2_adjustment


			print ("New Weight:")
			print (self.layer1.synaptic_weights)
			print (self.layer2.synaptic_weights)
			print ("--------------------------")

			print("================================================================")


	    # The neural network thinks.
	def think(self, inputs):
		output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
		output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
		print("layer2 sebelum Sigmoid")
		print(dot(output_from_layer1, self.layer2.synaptic_weights))
		return output_from_layer1, output_from_layer2

	def predict(self, inputs):
		output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
		output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
		print("layer2 sebelum Sigmoid")
		print(dot(output_from_layer1, self.layer2.synaptic_weights))
		return output_from_layer1, output_from_layer2

	# The neural network prints its weights
	def print_weights(self):
		print ("Layer 1 (2 neurons, each with 2 inputs): ")
		print (self.layer1.synaptic_weights)


		print ("Layer 2 (1 neuron, with 2 inputs):")
		print (self.layer2.synaptic_weights)


if __name__ == "__main__":

	#Seed the random number generator 
	random.seed(1)

	# Create layer 1 (2 neurons, each with 2 inputs)
	layer1 = NeuronLayer(2, 2)

	# Create layer 2 (1 neurons, each with 2 inputs)
	layer2 = NeuronLayer(1, 2)
	# Combine the layers to create a neural network
	neural_network = NeuralNetwork(layer1, layer2)

	print ("Stage 1) Random starting synaptic weights: ")
	neural_network.print_weights()

	training_set_inputs = array([[5.1, 3.8], [4.6, 3.2], [5.3, 3.7], [5.6, 3.0], [6.2, 2.2], [6.7, 3.0]])
	training_set_outputs = array([[0, 0, 0, 1, 1, 1]]).T

	# Train the neural network using the training set.
	# Do it 60,000 times and make small adjustments each time.
	neural_network.train(training_set_inputs, training_set_outputs, 4)

	print ("Stage 2) New synaptic weights after training: ")
	neural_network.print_weights()

	# Test the neural network with a new situation.
	print ("Stage 3) Considering a new situation [4.4, 3.2] -> ?: ")
	hidden_state, output = neural_network.think(array([4.4, 3.2]))
	print (output)
