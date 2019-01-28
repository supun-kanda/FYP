import h5py
import numpy as np
filename = 'file.hdf5'
f = h5py.File('PCA_activitynet_v1-3.hdf5', 'r')

# List all groups
#print('shape_is', f.keys.shape)
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])

print(np.array(f[a_group_key]))

import h5py
filename = 'PCA_activitynet_v1-3.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
print(list(f['data'].keys()))
#print(f['type'])
