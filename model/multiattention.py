from model import model_
import tensorflow as tf

def attention_dot_product(query,key,value, mask):
    qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = qk/tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(weights,value)
    return output

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units = d_model)
        self.key_dense = tf.keras.layers.Dense(units = d_model)
        self.value_dense = tf.keras.layers.Dense(units = d_model)
        self.dense = tf.keras.layers.Dense(units = d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape = (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0,2,1,3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'],inputs['key'],inputs['value'],inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)

        scaled_attention = tf.transpose(attention_dot_product(query, key, value, mask), perm=[0,2,1,3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        
        return outputs