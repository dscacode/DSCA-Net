import tensorflow as tf
import numpy as np
import os
import keras.backend as K
import keras
import datetime
from sklearn.metrics import jaccard_score, hamming_loss, label_ranking_average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import *
from keras.preprocessing import sequence
from keras.layers import *
from keras import *
from DSCA import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # print(inputs.shape)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, seqlen))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(seqlen, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    # print('attention(lambda):', print(a.shape))
    a = RepeatVector(input_dim)(a)
    # print('attention(repeatv):', print(a.shape))
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    # print('attention inputs,a_probs,output_attention_mul:', inputs.shape, a_probs.shape, output_attention_mul.shape)
    return output_attention_mul


def jaccard_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_true_expand = K.expand_dims(y_true, axis=0)
    y_pred_expand = K.expand_dims(y_pred, axis=-1)

    fenzi = K.dot(y_true_expand, y_pred_expand)
    fenmu_1 = K.sum(y_true, keepdims=True)
    fenmu_2 = K.ones_like(y_true_expand) - y_true_expand
    fenmu_2 = K.dot(fenmu_2, y_pred_expand)

    return K.mean((tf.constant([[1]], dtype=tf.float32) - (fenzi / (fenmu_1 + fenmu_2))), axis=-1)
    # return K.mean((fenzi / (fenmu_1 + fenmu_2)), axis=-1) s


def jaccard_score_approximation(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    one = K.ones_like(y_pred)
    zero = K.zeros_like(y_pred)

    y_pred = tf.where(y_pred < 0.5, x=zero, y=one)

    y_true_expand = K.expand_dims(y_true, axis=0)
    y_pred_expand = K.expand_dims(y_pred, axis=-1)

    a_and_b = K.dot(y_true_expand, y_pred_expand)
    a_or_b = K.sum(y_true, keepdims=True) + K.sum(y_pred, keepdims=True)
    a_or_b = a_or_b - a_and_b

    return a_and_b / a_or_b


def my_test(y_true, y_pred):
    y_true_after = []
    y_pred_after = []
    # y_true = np.array(y_true, dtype=int)
    sumequal0 = 0
    for ptrue, ppred in zip(y_true, y_pred):
        for dtrue, dpred in zip(ptrue, ppred):
            if sum(dtrue) > 0.0 or sum(dtrue) > 0:
                y_true_after.append(dtrue)
                y_pred_after.append(dpred)
    y_true_after = np.array(y_true_after)
    y_pred_after = np.array(y_pred_after)

    # print(y_true_after.shape, y_pred_after.shape)

    my_label_ranking_average_precision_score = label_ranking_average_precision_score(y_true=y_true_after, y_score=y_pred_after)
    my_micro_auc = roc_auc_score(y_true_after, y_pred_after, average='micro')

    y_pred_after[y_pred_after >= 0.5] = 1
    y_pred_after[y_pred_after < 0.5] = 0
    y_pred_after = np.array(y_pred_after, dtype=int)
    y_true_after = np.array(y_true_after, dtype=int)

    my_jaccard_score = jaccard_score(y_true=y_true_after, y_pred=y_pred_after, average='samples')
    my_hamming_loss = hamming_loss(y_true=y_true_after, y_pred=y_pred_after)
    my_micro_f1 = f1_score(y_true_after, y_pred_after, average='micro')


    return \
        my_jaccard_score, \
        my_hamming_loss, \
        my_label_ranking_average_precision_score, \
        my_micro_f1, \
        my_micro_auc


# Dsca
def get_dscagru(seqlen_, max_features_, seqx1_dim, seqx2_dim, seqx1_emb_dim, seqx2_emb_dim, tau_, rnnlay_units_, dlay_units_):
    inputs = Input(shape=(seqlen_, max_features_))
    hout = DscaGRU(units=rnnlay_units_, return_sequences=True,
                   seqx1_dim=seqx1_dim, seqx2_dim=seqx2_dim,
                   seqx1_emb_dim=seqx1_emb_dim,
                   seqx2_emb_dim=seqx2_emb_dim,
                   tau=tau_,
                   )(inputs)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout_final = Dense(400, activation=keras.activations.sigmoid)(hout)
    model = Model(inputs=inputs, outputs=hout_final, name='DSCA-GRU')
    return model


def get_dscalstm(seqlen_, max_features_, seqx1_dim, seqx2_dim, seqx1_emb_dim, seqx2_emb_dim, tau_, rnnlay_units_, dlay_units_):
    inputs = Input(shape=(seqlen_, max_features_))
    hout = DscaLSTM(rnnlay_units_, return_sequences=True,
                    seqx1_dim=seqx1_dim, seqx2_dim=seqx2_dim,
                    seqx1_emb_dim=seqx1_emb_dim,
                    seqx2_emb_dim=seqx2_emb_dim,
                    tau=tau_,
                    )(inputs)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout_final = Dense(400, activation=keras.activations.sigmoid)(hout)
    model = Model(inputs=inputs, outputs=hout_final, name='DSCA-LSTM')
    return model


def get_dscarnn(seqlen_, max_features_, seqx1_dim, seqx2_dim, seqx1_emb_dim, seqx2_emb_dim, tau_, rnnlay_units_, dlay_units_):
    inputs = Input(shape=(seqlen_, max_features_))
    hout = DscaRNN(rnnlay_units_, return_sequences=True,
                   seqx1_dim=seqx1_dim, seqx2_dim=seqx2_dim,
                   seqx1_emb_dim=seqx1_emb_dim,
                   seqx2_emb_dim=seqx2_emb_dim,
                   tau=tau_,
                   )(inputs)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout = Dense(dlay_units_, activation=keras.activations.relu)(hout)
    hout = Dropout(0.3)(hout)
    hout_final = Dense(400, activation=keras.activations.sigmoid)(hout)
    model = Model(inputs=inputs, outputs=hout_final, name='DSCA-RNN')
    return model


def train_model(model_, x_train_, y_train_, x_test_, y_test_, batch_size_, epochs_, patience_, verbose_):
    print(model_.name + ':')
    model_.compile('adam', loss=keras.losses.categorical_crossentropy, metrics=[jaccard_score_approximation])
    earlystop = keras.callbacks.EarlyStopping(monitor='val_jaccard_score_approximation', mode='max', patience=patience_,
                                              restore_best_weights=True)
    start_time = datetime.datetime.now()
    model_.fit(x_train_, y_train_,
               batch_size=batch_size_,
               epochs=epochs_,
               validation_data=[x_test_, y_test_],
               callbacks=[earlystop],
               verbose=verbose_
               )
    end_time = datetime.datetime.now()
    time = (end_time-start_time).seconds
    y_pred_ = model_.predict(x_test_)
    my_jaccard_score, my_hamming_loss, my_label_ranking_average_precision_score, \
    my_micro_f1, my_micro_auc = my_test(y_true=y_test_, y_pred=y_pred_)
    print(model_.name, time, [
        round(my_jaccard_score, 8),
        round(my_hamming_loss, 8),
        round(my_label_ranking_average_precision_score, 8),
        round(my_micro_f1, 8),
        round(my_micro_auc, 8)])
    print('JACCARD_SCORE:', my_jaccard_score)
    print('LABEL_RANKING_AVERAGE_PRECISION_SCORE:', my_label_ranking_average_precision_score)
    print('MICRO_F1:', my_micro_f1)
    print('MICRO_AUC:', my_micro_auc)


if __name__ == '__main__':
    maxlen = 100
    batch_size = 512
    seqlen = 35
    epochs = 200  # 100, 200, 300
    patience = 5
    verbose = 0
    rnnlay_units = 256
    dlay_units = 256
    kfolds = 5
    tau = 0.2
    print('Loading data...')
    x = np.load('/XXXXX/dualseqs_down_%d.npy' % maxlen, allow_pickle=True)
    y = np.load('/XXXXX/seqy_down_%d.npy' % maxlen, allow_pickle=True)
    max_features = x[0].shape[1]
    # Train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)

    print('\nPad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=seqlen, padding='post', value=0)
    x_test = sequence.pad_sequences(x_test, maxlen=seqlen, padding='post', value=0)
    y_train = sequence.pad_sequences(y_train, maxlen=seqlen, padding='post', value=0)
    y_test = sequence.pad_sequences(y_test, maxlen=seqlen, padding='post', value=0)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    lentr = len(x_train)
    lente = len(x_test)

    x_train = np.array(x_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    print('epochs:', epochs)
    print('batch_size:', batch_size)
    print('patience:', patience)
    # print('tau:', tau)
    print('x_train shape:',  x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    train_model(get_dscarnn(seqlen, max_features, 400, max_features - 400, 64, 64, tau, rnnlay_units, dlay_units),
                x_train, y_train, x_test, y_test, batch_size, epochs, patience, verbose)
    train_model(get_dscalstm(seqlen, max_features, 400, max_features - 400, 64, 64, tau, rnnlay_units, dlay_units),
                x_train, y_train, x_test, y_test, batch_size, epochs, patience, verbose)
    train_model(get_dscagru(seqlen, max_features, 400, max_features - 400, 64, 64, tau, rnnlay_units, dlay_units),
                x_train, y_train, x_test, y_test, batch_size, epochs, patience, verbose)


