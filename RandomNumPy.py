import numpy as np

# Generate a 2x3 array of random numbers between 0 and 0.1
random_array = np.random.rand(2, 3) * 0.1

# Print the generated array
print("arr1 =", random_array)
print('')  # Blank line

# Generate a 2x3 array of random numbers from a standard normal distribution
randn_array = np.random.randn(2, 3) * 0.1

# Print the generated array
print("arr2 =", randn_array)
print('')  # Blank line

# Generate and print 5 random integers between 2 and 15
for i in range(5):
    randint_array = np.random.randint(2, 16)
    print("arr3 =", randint_array)

abs=np.absolute(2+1j)
print(abs)

print(np.__file__)
