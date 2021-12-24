from tensorflow.keras.applications import xception
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

conv_base = xception.Xception(weights='imagenet', include_top=False)
conv_base.trainable = True

for layer in conv_base.layers[:-32]:
    layer.trainable = False

model = Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(200))
model.summary()