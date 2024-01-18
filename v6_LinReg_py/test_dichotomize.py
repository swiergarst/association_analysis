from sklearn.preprocessing import OneHotEncoder
import numpy as np

sample_input = np.array([1, 2, 0, 2, 0, 1])

ohe = OneHotEncoder(categories=[[0,1, 2]], sparse=False, drop='first')

mapped_arr = ohe.fit_transform(sample_input.reshape(-1, 1))

print(mapped_arr)
