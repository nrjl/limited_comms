import numpy as np
import itertools

def binary_tuples(w):
    return itertools.product((0, 1), repeat=w)

# Problem setup:
# Y = {y0,y1,...,yn} is the random variable we want to learn, the ACTIONS
# \Theta is the unobserved variable , the STATE
# X are conditionally independent observations of the state \Theta


# First, implement the medical diagnosis toy problem
# Theta are the conditions (binary tuples to represent conditions)
#Theta = list(binary_tuples(3))
Theta = [(1,0,0),(0,0,0),(1,1,1),(1,0,1),(0,1,0)]

# Y are treatments
Y = [{Theta[0],Theta[1]},{Theta[2],Theta[3]},{Theta[4]}]

