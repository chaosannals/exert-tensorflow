import tensorflow as tf
from tensorflow import keras

M_PATH = './premodel/captcha/'

model = keras.models.load_model(M_PATH)
