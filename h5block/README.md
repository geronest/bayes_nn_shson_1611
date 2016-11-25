# h5block: Blocked writing of hdf5 using python
* **h5block** supports almost the same interface as h5py.
* Currently supports, file and dataset only
* Some of dataset magic functions are not implemented yet.
* **But**, in most of the usual cases, you will not have inconveniences.

## Usage
```
import h5block
import numpy as np

fd = h5block.File('test.h5', 'w')
ds = fd.create_dataset('hi', (2,3), axis=0)
print ds

ds.extend(np.array([[1,2,3]]))

fd.close()
```
The above will first store (2,3)-shape zero matrix. Then, extend it one row with [1,2,3]. This can be done because we gave `axis=0`. It means 0-axis can be extended in the future. You can give `axis` array value like `axis=[0,1]`, which means it can be extended in both 0 and 1-axes. But, in this case, when you use `extend` function you have to give a `axis` as well like `ds.extend(some_arr, axis=1)`.

All the operations flush imediately. So you do not have to maintain the large data in your memory but append the newly created chunks to the disk.

### Auto-initialize

You can use `extend` without pre-defining the dataset. The `extend` function will auto-create the dataset as follows

```
import h5block
import numpy as np

fd = h5block.File('test.h5', 'w')
fd['hi'].extend(np.array([[1,2,3]]), axis=0)

fd.close()
```

However, in this case, you have to give `axis` keyword argument when you call `extend`.

