import numpy as np


rng = np.random.default_rng()

vals = abs(rng.standard_normal(size = (10, 30))).astype(object)

base_time = np.datetime64('2010-01-01')

rng_val = rng.standard_normal(size = len(vals[2,:]))

print(rng.standard_normal())

#for i in range(len(vals[2,:])):
#    vals[2,i] = base_time + int(rng.standard_normal())
#
#rint(vals)
vals[2, :] = np.array(base_time + rng.integers(0, 2000, size = len(vals[2,:]) )).astype(str)

print(vals[2,:])
print(base_time + rng.integers(9000, 10000, size = 10))