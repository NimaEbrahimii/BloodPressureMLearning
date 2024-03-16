import numpy as np

# Create two NumPy arrays
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# Print the arrays
print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

# Perform basic mathematical operations
print("\nAddition:")
print(np.add(array1, array2))

print("\nSubtraction:")
print(np.subtract(array1, array2))

print("\nMultiplication:")
print(np.multiply(array1, array2))

print("\nDivision:")
print(np.divide(array1, array2))

# Transpose an array
print("\nTranspose of Array 1:")
print(np.transpose(array1))

# Flatten an array
print("\nFlattened Array 1:")
print(array1.flatten())

# Reshape an array
print("\nReshaped Array 2:")
print(np.reshape(array2, (3, 2)))

# Concatenate arrays
print("\nConcatenated Arrays:")
print(np.concatenate((array1, array2), axis=0))  # Concatenate along rows

# Calculate the mean of an array
print("\nMean of Array 1:")
print(np.mean(array1))

# Calculate the sum of an array
print("\nSum of Array 2:")
print(np.sum(array2))



import numpy as np

x = np.int32((1024**3)*(2))
y = np.float64(3.14)
z = np.bool_(True)

print('x=',x,z,y)