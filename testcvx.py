import numpy as np
p = np.zeros(shape=(3,11))
print p
for t in range(0, 2):
	p[t+1] = p[t] + np.ones(11)
print p


