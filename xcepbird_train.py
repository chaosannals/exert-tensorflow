from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers, losses, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import xception
import tensorflow as tf
import os

IM_HEIGHT = 299
IM_WIDTH = 299
IM_PATH = './data/bird/'
TRAIN_DIR = IM_PATH + 'train'
VALIDATION_DIR = IM_PATH + 'valid'
TEST_DIR = IM_PATH + 'test'
BATCH_SIZE = 128
EPOCHS = 10


train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    directory=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IM_HEIGHT, IM_WIDTH),
    class_mode='categorical',
)

total_train = train_data_gen.n

validation_data_gen = validation_image_generator.flow_from_directory(
    directory=VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    target_size=(IM_HEIGHT,IM_WIDTH),
    class_mode='categorical',
)
total_validation = validation_data_gen.n

test_data_gen = test_image_generator.flow_from_directory(
    directory=TEST_DIR,
    target_size=(IM_HEIGHT, IM_WIDTH),
)

total_test = test_data_gen.n

conv_base = xception.Xception(weight='imagenet', include_top=False)
conv_base.trainable = True

for layer in conv_base.layers[:-32]:
    layer.trainable = False

model = Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(200))
model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss=losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    verbose=1,
)

history = model.fit(
    x=train_data_gen,
    steps_per_epoch=total_train,
    epochs=EPOCHS,
    validation_data=validation_data_gen,
    validation_steps=total_validation,
    callbacks=[early_stopping, reduce_lr],
)

history_dict = history.history
train_loss = history_dict['loss']