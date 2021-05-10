import tensorflow as tf            
import numpy as np

class MultiHeadRelativeAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, max_seq):
        super(MultiHeadRelativeAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}'
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)
        
        self.max_seq = max_seq
        self.E = self.add_weight('emb', shape=[self.max_seq, self.embed_dim // self.num_heads])
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'max_seq': self.max_seq,
        })
        return config

    def build(self, input_shape):
        self.len_q = input_shape[1]
        self.len_k = input_shape[1]
    
    @staticmethod
    def causal_attention_mask(n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        return tf.cast(m, dtype)
    
    @staticmethod
    def QE_mask(QE, dtype):
        mask = tf.sequence_mask(tf.range(QE.shape[-1] - 1, QE.shape[-1] - QE.shape[-2] - 1, -1), QE.shape[-1])

        mask = tf.logical_not(mask)
        return tf.cast(mask, dtype)

    def skewing(self, tensor):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]])
        relative_score = reshaped[:, :, 1:, :]

        return relative_score

    def relative_attention(self, query):
        QE = tf.einsum('bhld,md->bhlm', query, self.E)
        mask = self.QE_mask(QE, QE.dtype)
        QE = QE * mask
        return self.skewing(QE)

    def attention(self, query, key, value):
        relative_score = self.relative_attention(query)
        score = tf.matmul(query, key, transpose_b=True)

        score += relative_score
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        shape = tf.shape(scaled_score)
        dim_dest, dim_src = shape[2], shape[3]
        attention_mask = self.causal_attention_mask(
            dim_dest, dim_src, scaled_score.dtype
        )
        attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
        scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, self.max_seq, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate, max_seq):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.max_seq = max_seq
        self.att = MultiHeadRelativeAttention(embed_dim, num_heads, max_seq)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation='relu'), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'max_seq': self.max_seq
        })
        return config

    def __call__(self, inputs):
        attention_output = self.att(inputs)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seq, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_seq, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_seq': self.max_seq,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

    def __call__(self, x):
        max_seq = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_seq, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
