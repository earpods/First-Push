
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def sigmoid ( z ):
	return 1 / ( 1 + np.exp( -z ) )

def initialize( data ):
	w = np.zeros( shape = ( data.shape[ 0 ], 1 ) )
	b = 0
	return w, b

def propagate( w, b, X, Y ):
	
	m    = float( X.shape[ 1 ] )
	n    = float( X.shape[ 0 ] )

	A    = sigmoid( np.dot( w.T, X ) + b ).T
	cost = ( - 1 / m ) * np.sum( 
		np.dot( Y, np.log( A ) ) + np.dot( Y, np.log( 1 - A )) 
		)
	#print A.shape
	#print (Y).shape
	# print A
	# print Y
	# print cost
	dz   = A - Y.T 
	#print dz, m
	#print dz.shape
	dw   = 1/m * np.dot( X, dz ) 
	#print dw 
	db   = 1/m * np.sum( dz )
	#print dw.shape
	Gradient  = { "dw" : dw, 
		"db" : db }

	return Gradient, cost

def optimize( w, b, X, Y, Iterations, learning_rate ):

	for I in range( Iterations ):
		Gradient, cost = propagate( w, b, X, Y )

		w -= learning_rate * Gradient[ "dw" ]
		b -= learning_rate * Gradient[ "db" ]
		print Gradient[ "db" ] 
	params = { "w" : w,
		"b" : b }

	return params

def predict( w, b, X ):
	predictions = np.dot( w.T, X ) + b;
	results 	= []
	print predictions.shape
	for elem in predictions.flatten() :
		if( elem > 0.5 ):
			results.append( 1 )
		else:
			results.append( 0 )
	return results
mnist = input_data.read_data_sets( "MNIST_data/", reshape = True, one_hot = False)

train_pixels = np.array( mnist.train.images )
all_labels 	 = np.array( mnist.train.labels )
pixels 		 = np.array( train_pixels )

train_data   = []
train_labels = []

for ( index, val ) in enumerate( all_labels ):
	if( val == 0 or val == 1 ):
		train_data.append( train_pixels[ index ] )
		train_labels.append( all_labels[ index ] )
test_labels  = np.array( [ train_labels[ 10000: 11000 ] ])
test_data 	 = np.array( train_data[ 10000: 11000 ])
train_data   = np.array( train_data[ 1: 10000 ] )
train_labels = np.array( [ train_labels[ 1: 10000 ] ] )

print train_pixels.shape
print train_data.shape
print train_labels.shape
assert( train_data.shape, train_labels.shape )

# print ( "Number of training examples: m_train = " + str( train_data.shape[ 0 ] ) )
# print ( "train_element size: " + str( train_data[ 0 ].shape[ 0 ]))


#Get Started with the process 
w, b = initialize( train_data )
#print sigmoid( np.array( [ 1,2,3 ] ) )
#print optimize( np.array( [ [ 1.1 ], [ 2 ], [ 3 ], [ 4 ]  ] ), 2, np.array( [ [ 2,3,4,5 ], [ 32.2,23, 4,4  ] ] ).T, np.array( [ [ 1 , 0 ] ] ), 10, 0.001 )
train_data = train_data.T
test_data  = test_data.T
#print propagate( np.zeros( shape = ( 4, 1 ) ), 2, np.array( [ [ 1, 12 ], [ 2, 3 ], [ 2, 13 ], [ 4, 8  ]  ]), np.array( [ [ 0, 1 ] ] ) )
w, b   = initialize( train_data )
params = optimize( w, b, train_data, train_labels, 100 , 0.05 )
w = params[ "w" ]
b = params[ "b" ]
#print " results "
#print test_data.shape
results = predict( w, b, test_data )
#print results

print np.mean( np.abs( test_labels.flatten() - results )* 100 )





