import numpy as np
import os
import time
from CH_load_data import load_cholec_data

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



base_dir = "data/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"

img_data, label = load_cholec_data(image_dir, label_dir)

num_classes = label.shape[1]

#Shuffle the dataset
x,y = shuffle(img_data,label, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(img_data, label, test_size=0.2, random_state=2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-8]:
	layer.trainable = False

#custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


t=time.time()
#	t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=32, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


