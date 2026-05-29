import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
