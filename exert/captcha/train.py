from importlib import import_module
from loguru import logger
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

CaptchaAssets = import_module('data').CaptchaAssets
CaptchaCnnModel = import_module('cnn').CaptchaCnnModel

EPOCHS = 5
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 202
MAX_CAPTCHA = 6
CHARSET_LENGTH = len('2345678abcdefhijkmnpqrstuvwxyz')

model = CaptchaCnnModel(
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_CAPTCHA,
    CHARSET_LENGTH,
)
assets = CaptchaAssets('captcha')


def loss(x, y):
    y1 = tf.reshape(y, (-1, MAX_CAPTCHA * CHARSET_LENGTH))
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x,
            labels=tf.cast(y1, tf.float32),
        )
    )


optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)


# @tf.function
def train_step(x, y):
    '''
    '''

    with tf.GradientTape() as tape:
        r1 = model(x)
        r2 = loss(r1, y)
    g = tape.gradient(r2, model.trainable_variables)
    optimizer.apply_gradients(zip(g, model.trainable_variables))
    return r2

def train_accuracy(x, y):
    '''
    '''

    r1 = model(x)
    y1 = tf.reshape(y, (-1, MAX_CAPTCHA * CHARSET_LENGTH))
    max_idx_p = tf.argmax(r1, -2)
    max_idx_l = tf.argmax(y1, -2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32)
    )
    return accuracy



def train():
    '''
    训练。
    '''

    epoch = 0
    while True:
        x, y = assets.get_batch()
        loss = train_step(x, y)
        epoch += 1
        if epoch % 10 == 0:
            tx, ty = assets.get_batch()
            acc = train_accuracy(tx, ty)
            logger.info(f'accuracy: {acc}')
            if acc > 0.7:
                
                break
        logger.info(f'epoch: {epoch} loss: {loss}')


if __name__ == '__main__':
    train()
