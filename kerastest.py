# Experiment to see if Keras can figure out if the numbers correspond to the
# formula z=3x+2y+5
import tensorflow
import numpy
# fix random seed for reproducibility
numpy.random.seed(99)
# load csv file
dataset = numpy.loadtxt("formula.csv", delimiter=",")
# split into input (X) and output (Y) variables
# see if the neural net can figure out the formula z=3x+2y+5 (col 1 = x, col 2 = y, col 3 = z)
X = dataset[:,0:2]
Y = dataset[:,2]

# create model
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(16, input_dim=2, activation='relu'))
model.add(tensorflow.keras.layers.Dense(8, activation='relu'))
model.add(tensorflow.keras.layers.Dense(1, activation='linear'))
# Compile model
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mean_absolute_error', optimizer='adam')
#model.compile(loss='mean_absolute_error', optimizer='sgd')
#model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
# Fit the model
model.fit(X, Y, epochs=3000)
# Let's see if the Neural Network arrived at the correct conclusion (105)
Xnew = numpy.array([[20,20]])
predictions = model.predict(Xnew)
print(predictions[0])
# Let's see if the Neural Network arrived at the correct conclusion (55)
Xnew = numpy.array([[10,10]])
predictions = model.predict(Xnew)
print(predictions[0])