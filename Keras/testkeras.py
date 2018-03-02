'''
use keras
'''
# %% Part-1
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

X=np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y=np.array([[0], [0], [0], [1]], dtype=np.float32)

#Creata a Sequential model
model=Sequential()
model.add(Dense(32, input_dim=X.shape[0])) # 1st layer : 32 nodes with the same input shape as X
model.add(Activation('softmax'))
model.add(Dense(1)) # 2nd Layer
model.add(Activation('sigmoid')) # Add the sigmoid Activation layer
print(model.summary)
