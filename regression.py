import numpy as np

x1 = np.array([1,5,8])
x2 = np.array([3,2,4])
y = np.array([5,4,7])
n = x1.shape[0]

# print(x1.sum())
# print(x1.shape[0])
# print(np.square(x1))
# print(np.square(x2))
# print(np.square(y))
# print(np.multiply(x1,x2))
# print(np.multiply(x1,y))
# print(np.multiply(x2,y))

sumX1 = x1.sum()
sumX2 = x2.sum()
sumY = y.sum()

sumSquareX1 = np.square(x1).sum()
sumSquareX2 = np.square(x2).sum()
sumSquareY = np.square(y).sum()
sumSquareX1X2 = np.multiply(x1,x2).sum()
sumSquareX1Y = np.multiply(x1,y).sum()
sumSquareX2Y = np.multiply(x2,y).sum()

sigSumSquareX1 = sumSquareX1 - ((sumX1 ** 2)/ n)
sigSumSquareX2 = sumSquareX2 - ((sumX2 ** 2) / n)
sigSumSquareY = sumSquareY - ((sumY ** 2) / n)
sigSumSquareX1X2 = sumSquareX1X2 - ((sumX1 * sumX2) / n)
sigSumSquareX1Y = sumSquareX1Y - ((sumX1 * sumY) / n)
sigSumSquareX2Y = sumSquareX2Y - ((sumX2 * sumY) / n)

print("sigSumSquareX1 "+str(sigSumSquareX1))
print("sigSumSquareX2 "+str(sigSumSquareX2))
print("sigSumSquareY "+str(sigSumSquareY))
print("sigSumSquareX1X2 "+str(sigSumSquareX1X2))
print("sigSumSquareX1Y "+str(sigSumSquareX1Y))
print("sigSumSquareX2Y "+str(sigSumSquareX2Y))

b1 = ((sigSumSquareX2 * sigSumSquareX1Y) - (sigSumSquareX2Y * sigSumSquareX1X2)) / ((sigSumSquareX1 * sigSumSquareX2) - (sigSumSquareX1X2 ** 2))
b2 = ((sigSumSquareX1 * sigSumSquareX2Y) - (sigSumSquareX1Y * sigSumSquareX1X2)) / ((sigSumSquareX1 * sigSumSquareX2) - (sigSumSquareX1X2 ** 2))
a = (((sumY) - (b1 * sumX1) - (b2 * sumX2)) / n)

print("a "+str(a))
print("b1 "+str(b1))
print("b2 "+str(b2))

learning_rate = 0.2

for i in range(2):
	print("================================================================")
	print("               EPOCH" + str(i) + "               ")
	print("================================================================")
	sumError = 0.0
	for j in range(x1.shape[0]):
		print("data ke= "+ str(j), end = " ; ")
		output = a + (b1 * x1[j]) + (b2 * x2[j])
		error = y[j] - output
		sumError = sumError + error**2
		print("error= "+ str(error), end = " ; ")
		print("hasil_prediksi= "+ str(output))
		#print(error)
		new_a = a + (learning_rate * error)
		new_b1 = b1 + (learning_rate * error * x1[j])
		new_b2 = b2 + (learning_rate * error * x2[j])

		a = new_a
		b1 = new_b1
		b2 = new_b2

		print("new weight")
		print("a= " + str(new_a))
		print("b1= " + str(new_b1))
		print("b2= " + str(new_b2))
		print(" ")
	print("sum error: "+ str (sumError))
	print("================================================================")