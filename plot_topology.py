import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ast

matplotlib.rcParams.update({'font.size': 22})

x = np.array([0,1,2,3,4,5,6])

my_xticks = ['Input','Filter','ReLU','Filter','ReLU','Filter','ReLU']
plt.xticks(x, my_xticks)
plt.ylabel('Separability (0 = perfectly separable, 50 = perfectly inseparable)')

with open('results.txt') as f:
    for line in f:
        y = ast.literal_eval(line)
        plt.plot(x, y, color='cornflowerblue', linewidth=3)

plt.show()
