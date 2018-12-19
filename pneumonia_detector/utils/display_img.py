
import sys
import pydicom
import numpy as np
from matplotlib import pyplot as plt

path = sys.argv[1]
img = pydicom.dcmread(path).pixel_array.astype(np.uint8)
print(img.shape)
s = plt.imshow(img)
plt.show(s)
