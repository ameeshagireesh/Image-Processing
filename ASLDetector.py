import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

train = pd.read_csv('data/sign_mnist_train.csv')
test = pd.read_csv('data/sign_mnist_test.csv')

# train_data = 