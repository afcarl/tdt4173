#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from em import DataManager

mus = [-0.5766685631700593, 2.978370252317221]
sigma = 1.0

x = np.array(DataManager().data)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# overlay gaussian lines
for mu in mus:
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Value')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram}$')
plt.axis([-4, 6, 0, 0.5])
plt.grid(True)

plt.show()
