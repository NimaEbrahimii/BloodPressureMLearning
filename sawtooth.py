import numpy as np
import matplotlib.pyplot as plt

# Define the function
def x(t):
    return 2 * (t - np.floor(t + 0.5))

# Create an array of t values
t_values = np.linspace(-2, 2, 1000)

# Calculate x(t) values
x_values = x(t_values)

# Plot the function
plt.figure(figsize=(8, 4))
plt.plot(t_values, x_values, label=r"$x(t) = 2 \left( t - \left\lfloor t + \frac{1}{2} \right\rfloor \right)$")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Sawtooth Wave")
plt.grid(True)
plt.legend()
plt.show()



