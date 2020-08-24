import tensorflow as tf
import tensorflow_hub as hub


class IdentityModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.identity(inputs)


class USEModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(self.embedding(tf.squeeze(inputs)))


identity_model = IdentityModel()
identity_model.compile(loss='binary_crossentropy', optimizer='sgd')
identity_model.fit([0.0, 1.0], [0.0, 1.0])
identity_model.save('export/identity/0')

use_model = USEModel()
use_model.compile(loss='binary_crossentropy', optimizer='sgd')
use_model.fit(['a sentence', 'b sentence'], [0.0, 1.0])
use_model.save('export/use/0')