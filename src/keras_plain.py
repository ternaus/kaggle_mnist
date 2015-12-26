from __future__ import division
'''
This is Neural Network with 2 hidden and one dropout layers that
gives 0.91643
'''

import pandas as pd
import numpy as np
np.random.seed(42)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

print 'reading train'
train = pd.read_csv('../data/train.csv')
X_train = train.drop('label', 1).values.astype('float32')

y_train = train['label'].values
y_train = np_utils.to_categorical(y_train)

print 'reading test'
test = pd.read_csv('../data/test.csv')
X_test = test.values.astype('float32')

print 'normalizing data'
X_train /= 255
X_test /= 255

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

print 'defining model'
model = Sequential()
model.add(Dense(200, input_dim=input_dim))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print 'compiling model'
model.compile(loss='categorical_crossentropy', optimizer='sgd')

print 'fitting'
model.fit(X_train, y_train,
  nb_epoch=10, 
  validation_split=0.2, 
  show_accuracy=True, 
  batch_size=16,
  verbose=1
)
print 'predicting'
prediction = model.predict_classes(X_test)

print 'saving'
num_images = test.shape[0]
submission = pd.DataFrame()
submission['ImageId'] = range(1, num_images + 1)
submission['Label'] = prediction
submission.to_csv('prediction_plain.csv', index=False)