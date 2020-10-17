import matplotlib.pyplot as plt
import numpy as np
import pdb

x = np.linspace(-1, 1, 1000)

cutoff = 0.3
satslope = 10
y = np.piecewise(x,
        [x < -cutoff, np.logical_and(x >= -cutoff, x < cutoff),
         x >= cutoff],
        [lambda x: x * satslope + (satslope - 1) * cutoff,
         lambda x: x,
         lambda x: x * satslope - (satslope - 1) * cutoff])

plt.plot(x, y)
plt.show()
