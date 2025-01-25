from keras.layers import Attention, MaxPooling1D, Dense, Dropout, Concatenate, Bidirectional, LSTM, Embedding, Input
from keras.models import Model
import keras.backend as K
from keras import initializers
from keras.src.engine.input_spec import InputSpec
from keras.src.engine.base_layer import Layer
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras
import numpy as np
import tensorflow as tf

MAX_LEN_mi = 30
MAX_LEN_M = 4000
EMBEDDING_DIM = 72

kmers = 6
name='npy/'
# if kmers == 4:
#     NB_WORDS = 257
#     lncRnaembedding_matrix = np.load('lnc4mers.npy')
#     miRnaembedding_matrix = np.load('mi4mers.npy')
if kmers == 6:
    NB_WORDS = 4096
    mRnaembedding_matrix = np.load(name+'m_fast_64.npy',allow_pickle=True)
    miRnaembedding_matrix = np.load(name+'mi_fast.npy',allow_pickle=True)
# elif kmers == 5:
#     NB_WORDS = 1025
#     lncRnaembedding_matrix = np.load('./processData/5mer/lnc5mers.npy')
#     miRnaembedding_matrix = np.load('./processData/5mer/mi5mers.npy')
# elif kmers == 3:
#     NB_WORDS = 65
#     lncRnaembedding_matrix = np.load('3mrna.npy')
#     miRnaembedding_matrix = np.load('3mirna.npy')



class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall


from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Concatenate, Embedding
from keras.models import Model



from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Concatenate, Embedding, Flatten, Reshape, Attention
from keras.models import Model





from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Concatenate
from keras.models import Model
from keras.layers import MultiHeadAttention

def get_model(len_behavior1, len_behavior2):

    miRna = Input(shape=(MAX_LEN_mi,))
    mRna = Input(shape=(MAX_LEN_M,))
    behavior1 = Input(shape=(len_behavior1,))
    behavior2 = Input(shape=(len_behavior2,))

    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[miRnaembedding_matrix], trainable=True)(miRna)
    emb_m = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[mRnaembedding_matrix], trainable=True)(mRna)


    miRna_conv = Conv1D(filters=36, kernel_size=3, padding="same", activation='relu')(emb_mi)
    miRna_conv = MaxPooling1D(pool_size=2)(miRna_conv)  
    miRna_conv = Dropout(0.5)(miRna_conv)

    mRna_conv = Conv1D(filters=36, kernel_size=3, padding="same", activation='relu')(emb_m)
    mRna_conv = MaxPooling1D(pool_size=5)(mRna_conv)  
    mRna_conv = Dropout(0.5)(mRna_conv)


    miRna_lstm = Bidirectional(LSTM(36, return_sequences=True))(miRna_conv)
    mRna_lstm = Bidirectional(LSTM(36, return_sequences=True))(mRna_conv)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=72)(miRna_lstm, mRna_lstm)


    attention_output = GlobalAveragePooling1D()(attention_output)


    combined_features = Concatenate()([attention_output, behavior1, behavior2])
    # combined_features = Average()([attention_output, behavior1, behavior2])
    # combined_features = Average()([attention_output, behavior1, behavior2])


    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)



    output = Dense(1, activation='sigmoid')(x)


    model = Model(inputs=[miRna, mRna, behavior1, behavior2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

