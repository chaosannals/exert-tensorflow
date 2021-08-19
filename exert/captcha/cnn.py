import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger


class CaptchaCnnModel(keras.Model):
    '''
    验证码识别模型
    '''

    def __init__(self, image_height, image_width, max_captcha, charset_length, **kws):
        '''
        '''

        super().__init__(dtype=tf.float32)
        self.lv1 = CaptchaCnnLv1Layer(
            image_height,
            image_width,
            **kws
        )
        self.lv2 = CaptchaCnnLv2Layer(**kws)
        self.lv3 = CaptchaCnnLv3Layer(**kws)
        self.lvd = CaptchaCnnDenseLayer(
            math.ceil(image_height / 8),
            math.ceil(image_width / 8),
            **kws
        )
        self.lvo = CaptchaCnnOutputLayer(
            max_captcha,
            charset_length,
            **kws
        )

    def call(self, x):
        '''
        '''

        r1 = self.lv1(x)
        r2 = self.lv2(r1)
        r3 = self.lv3(r2)
        r4 = self.lvd(r3)
        return self.lvo(r4)


class CaptchaCnnLv1Layer(layers.Layer):
    '''
    训练第一层
    '''

    def __init__(self, image_height, image_width, **kws):
        '''
        初始化数据
        '''

        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.w_alpha = kws['w_alpha'] if 'w_alpha' in kws else 0.01
        self.b_alpha = kws['b_alpha'] if 'b_alpha' in kws else 0.1
        self.droprate = kws['droprate'] if 'droprate' in kws else 0.5
        self.w = tf.Variable(self.w_alpha * tf.random.normal((5, 5, 4, 32)))
        self.b = tf.Variable(self.b_alpha * tf.random.normal((32,)))

    def build(self, input_shape):
        '''
        '''
        logger.info(f'lv1: {input_shape}')
        self.w = tf.Variable(self.w_alpha * tf.random.normal((5, 5, 4, 32)))
        self.b = tf.Variable(self.b_alpha * tf.random.normal((32,)))

    def call(self, x):
        '''
        '''

        r1 = tf.nn.conv2d(x, self.w, strides=(1, 1, 1, 1), padding='SAME')
        r2 = tf.nn.bias_add(r1, self.b)
        r3 = tf.nn.relu(r2)
        r4 = tf.nn.max_pool(
            r3,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME'
        )
        return tf.nn.dropout(r4, self.droprate)


class CaptchaCnnLv2Layer(layers.Layer):
    '''
    训练第二层
    '''

    def __init__(self, **kws):
        '''
        初始化数据
        '''

        super().__init__()
        self.w_alpha = kws['w_alpha'] if 'w_alpha' in kws else 0.01
        self.b_alpha = kws['b_alpha'] if 'b_alpha' in kws else 0.1
        self.droprate = kws['droprate'] if 'droprate' in kws else 0.5

    def build(self, input_shape):
        '''
        '''
        logger.info(f'lv2: {input_shape}')
        self.w = tf.Variable(self.w_alpha * tf.random.normal((5, 5, 32, 64)))
        self.b = tf.Variable(self.b_alpha * tf.random.normal((64,)))

    def call(self, x):
        '''
        '''

        r1 = tf.nn.conv2d(x, self.w, strides=(1, 1, 1, 1), padding='SAME')
        r2 = tf.nn.bias_add(r1, self.b)
        r3 = tf.nn.relu(r2)
        r4 = tf.nn.max_pool(
            r3,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME'
        )
        return tf.nn.dropout(r4, self.droprate)


class CaptchaCnnLv3Layer(layers.Layer):
    '''
    训练第三层
    '''

    def __init__(self, **kws):
        '''
        初始化数据
        '''

        super().__init__()
        self.w_alpha = kws['w_alpha'] if 'w_alpha' in kws else 0.01
        self.b_alpha = kws['b_alpha'] if 'b_alpha' in kws else 0.1
        self.droprate = kws['droprate'] if 'droprate' in kws else 0.5

    def build(self, input_shape):
        '''
        '''

        logger.info(f'lv3: {input_shape}')
        self.w = tf.Variable(self.w_alpha * tf.random.normal((5, 5, 64, 64)))
        self.b = tf.Variable(self.b_alpha * tf.random.normal((64,)))

    def call(self, x):
        '''
        '''

        r1 = tf.nn.conv2d(x, self.w, strides=(1, 1, 1, 1), padding='SAME')
        r2 = tf.nn.bias_add(r1, self.b)
        r3 = tf.nn.relu(r2)
        r4 = tf.nn.max_pool(
            r3,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME'
        )
        return tf.nn.dropout(r4, self.droprate)


class CaptchaCnnDenseLayer(layers.Layer):
    '''
    收敛层
    '''

    def __init__(self, input_height, input_width, **kws):
        '''
        '''

        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.w_alpha = kws['w_alpha'] if 'w_alpha' in kws else 0.01
        self.b_alpha = kws['b_alpha'] if 'b_alpha' in kws else 0.1
        self.droprate = kws['droprate'] if 'droprate' in kws else 0.5

    def build(self, input_shape):
        '''
        '''

        logger.info(f'dense: {input_shape}')
        self.w = tf.Variable(
            self.w_alpha * tf.random.normal(
                (self.input_height * self.input_width * 64, 1024)
            )
        )
        self.b = tf.Variable(self.b_alpha * tf.random.normal((1024,)))

    def call(self, x):
        '''
        '''

        r = tf.reshape(x, (-1, self.w.get_shape().as_list()[0]))
        r1 = tf.matmul(r, self.w)
        r2 = tf.add(r1, self.b)
        r3 = tf.nn.relu(r2)
        return tf.nn.dropout(r3, self.droprate)


class CaptchaCnnOutputLayer(layers.Layer):
    '''
    输出层
    '''

    def __init__(self, max_captcha, charset_length, **kws):
        '''
        '''

        super().__init__()
        self.max_captcha = max_captcha
        self.charset_length = charset_length
        self.w_alpha = kws['w_alpha'] if 'w_alpha' in kws else 0.01
        self.b_alpha = kws['b_alpha'] if 'b_alpha' in kws else 0.1
        self.droprate = kws['droprate'] if 'droprate' in kws else 0.5

    def build(self, input_shape):
        '''
        '''

        logger.info(f'output: {input_shape}')
        self.w = tf.Variable(
            self.w_alpha * tf.random.normal(
                (1024, self.max_captcha * self.charset_length)
            )
        )
        self.b = tf.Variable(
            self.b_alpha * tf.random.normal((self.max_captcha * self.charset_length,)))

    def call(self, x):
        '''
        '''

        r = tf.matmul(x, self.w)
        return tf.add(r, self.b)
