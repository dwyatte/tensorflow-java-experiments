import os
import tensorflow as tf

inputs = tf.placeholder(tf.int32, [None, 1])
outputs = tf.identity(inputs)

inputs_info = tf.saved_model.utils.build_tensor_info(inputs)
outputs_info = tf.saved_model.utils.build_tensor_info(outputs)

signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={tf.saved_model.PREDICT_INPUTS: inputs_info},
        outputs={tf.saved_model.PREDICT_OUTPUTS: outputs_info},
        method_name=tf.saved_model.PREDICT_METHOD_NAME
    )
)

with tf.Session() as sess:
    builder = tf.saved_model.builder.SavedModelBuilder('export/tf1/0')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.SERVING],
        signature_def_map={tf.saved_model.PREDICT_METHOD_NAME: signature}
    )
    builder.save()

