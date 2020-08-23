import tensorflow as tf

class IdentityModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.identity(inputs)

model = IdentityModel()
model._set_inputs([0.0])

model.save('export/tf2/0')