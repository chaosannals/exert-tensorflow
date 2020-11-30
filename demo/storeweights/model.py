import tensorflow as tf
from tensorflow import keras


class WeightModel:
    def __init__(self):
        self.checkpoint_path = "premodel/demo-storeweight/cp.ckpt"
        self.model = keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        (train_images, train_labels), (test_images,
                                       test_labels) = keras.datasets.mnist.load_data()

        self.train_labels = train_labels[:1000]
        self.test_labels = test_labels[:1000]

        self.train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
        self.test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    def load_weight(self):
        # 加载权重
        self.model.load_weights(self.checkpoint_path)

    def evaluate(self):
        loss, acc = self.model.evaluate(
            self.test_images,
            self.test_labels,
            verbose=2
        )
        print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    def summary(self):
        self.model.summary()

    def fit(self):
        '''
        '''
        # 创建一个保存模型权重的回调
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        # 使用新的回调训练模型
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=10,
            validation_data=(
                self.test_images,
                self.test_labels
            ),
            callbacks=[cp_callback], # 通过回调训练
        )
