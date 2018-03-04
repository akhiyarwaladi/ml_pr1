# Make a prediction with weights
def predict(row, weights):
	#print(len(row))
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	#print (activation)
	return 1.0 if activation > 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	print(len(train[0]))
	# inisialisasi weight
	weights[0] = 0 # bias
	weights[1] = 0.2 # w1
	weights[2] = 0 # w2
	weights[3] = 0 # w3
	for epoch in range(n_epoch):
		print("================================================================")
		print("               EPOCH" + str(epoch) + "               ")
		print("================================================================")
		sum_error = 0.0
		for row in train:
			print("row data: "+str(row))
			prediction = predict(row, weights)
			print("prediction: "+str(prediction), end=" ")
			error = row[-1] - prediction
			print("error: " +str(error))
			sum_error += error**2
			# bobot tidak kita update karena soal tidak ada bobot, namun bias tetap ditampilkan di hasil tapi selalu 0					
			#weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
			print(["w1", "w2", "w3"], end=" = ")
			print(weights[1:])
			print(" ")
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		print("================================================================")
		print(" ")
		if(sum_error == 0.0):
			break
	return weights


# data dummy hanya untuk testing
# Calculate weights
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]

dataset = [[1,0,1,0],
		[0,0,0,0],
		[0.5,0,1,1],
		[1,1,0,1],
		[0,1,0.5,1]]


l_rate = 0.2
n_epoch = 100
weights = train_weights(dataset, l_rate, n_epoch)
print("Weight akhir " + str(weights))

# mulai prediksi data baru
data1 = [0.5,1,1] #data andi
data2 = [1,0,0] #data budi

hasil1 = predict(data1, weights)
hasil2 = predict(data2, weights)

print("Andi diprediksi " + str(hasil1))
print("Budi diprediksi " + str(hasil2))