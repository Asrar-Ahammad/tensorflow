import tensorflow as tf

# For GPU optimization
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Initializing a tensor
x = tf.constant(4, shape = (1,1),dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]]) # To create a 2d array
x = tf.ones((3,3)) # To create an array containing ones.Repalce ones with zeros for zeroes in matrix. Replace it with 'eye' for identity matrix
x = tf.random.normal((3,3),mean=0,stddev=1) # For genetaring a normal/standard deviation
x = tf.random.uniform((1,3),minval=0,maxval=1) # For generating a uniform distribution
x = tf.range(start=1,limit=10,delta=2) # For generating a range of values
x = tf.cast(x,dtype=tf.float64) # For converting the data type of the variable
print(x)

# Mathematical operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y) # Addition
z = tf.subtract(x,y)
z = tf.divide(x,y)
z = tf.multiply(x,y)
z = tf.tensordot(x,y,axes = 1) # dot product of tensors. first multiply same indices and add total

# Matrix operations
x = tf.random.normal((3,2))
y = tf.random.normal((2,3))

z = tf.matmul(x,y) # or x@y
print(z)

# Indexing 
x = tf.constant([1,2,3,4,5])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])
print(x[::-1])
# To print specific indices
indices = tf.constant([0,3])
x = tf.gather(x,indices)
print(x)
# For matrix
x = tf.constant([[1,2],
                [3,4],
                [7,8]])
print(x[0,:])
print(x[0:2,:])

# Reshaping of tensors
x = tf.range(9)
print(x)
x = tf.reshape(x,(3,3))
print(x)
x = tf.transpose(x,perm=[1,0])
print(x)