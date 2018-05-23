import math
import numpy as np

# Declare a vector that will carry the Input values
z = [4.0, 8.0, 12.0, 16.0, 4.0, 8.0, 4.0]
z = [i/4.0 for i in z]

# Example number 1 using the math module
z_exp = [math.exp(i) for i in z]
# print(round(i,2) for i in z_exp)

sum_z_exp = sum(z_exp)
softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print("Softmax with Math: ", softmax)


# Example number 2 using numpy module
softmax2 = lambda x : np.exp(x)/np.sum(np.exp(x))
print(softmax2(z))