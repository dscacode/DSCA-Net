# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import warnings
import keras
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import disable_tracking
from keras.engine.base_layer import InputSpec
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces
from keras.layers.recurrent import *


class DscaRNNCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        super(DscaRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, seqx1_emb_dim + seqx2_emb_dim)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        self.seqx1_dim = seqx1_dim
        self.seqx2_dim = seqx2_dim
        self.seqx1_emb_dim = seqx1_emb_dim
        self.seqx2_emb_dim = seqx2_emb_dim
        self.tau = tau

    def build(self, input_shape):
        margex_dim = self.seqx1_emb_dim + self.seqx2_emb_dim
        self.embed_weight_1 = self.add_weight(shape=(self.seqx1_dim, self.seqx1_emb_dim),
                                              name='embed_weight_1',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.embed_weight_2 = self.add_weight(shape=(self.seqx2_dim, self.seqx2_emb_dim),
                                              name='embed_weight_2',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_w = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_u = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_v = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_w = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_u = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_v = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.crohis_weight_w = self.add_weight(shape=(margex_dim, self.units),
                                               name='crohis_weight_w',
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)

        self.crohis_bias = self.add_weight(shape=(self.units,),
                                           name='crohis_bias',
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)

        self.crohis_decomp_weight = self.add_weight(shape=(margex_dim, margex_dim),
                                                    name='crohis_decomp_weight',
                                                    initializer=self.kernel_initializer,
                                                    regularizer=self.kernel_regularizer,
                                                    constraint=self.kernel_constraint)

        self.crohis_decomp_bias = self.add_weight(shape=(margex_dim,),
                                                  name='crohis_decomp_bias',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)

        self.kernel = self.add_weight(shape=(margex_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        # print(states)
        crohis_tm = states[1]

        seqx1 = inputs[:, :self.seqx1_dim]
        seqx2 = inputs[:, self.seqx1_dim:]

        embseqx1 = K.dot(seqx1, self.embed_weight_1)
        embseqx2 = K.dot(seqx2, self.embed_weight_2)

        # cross_att_1to2 = K.softmax(K.tanh(
        #     K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
        #     K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
        #                            K.dot(embseqx2, self.cross_attention_weight_1to2_v))
        #
        # #print('cross_att_1to2:', cross_att_1to2)
        # cross_att_2to1 = K.softmax(K.tanh(
        #     K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
        #     K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
        #                            K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        # #print('cross_att_2to1:', cross_att_2to1)

        cross_att_1to2 = K.softmax(K.tanh(
            K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
            K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
                                   K.dot(embseqx2, self.cross_attention_weight_1to2_v))

        # print('cross_att_1to2:', cross_att_1to2)
        cross_att_2to1 = K.softmax(K.tanh(
            K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
            K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
                                   K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        # print('cross_att_2to1:', cross_att_2to1)

        cross_att = K.concatenate([cross_att_1to2, cross_att_2to1])

        # print('cross_att:', cross_att)

        inputs = K.concatenate([embseqx1 * cross_att_1to2, embseqx2 * cross_att_2to1])

        # print('inputs:', inputs)

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(prev_output),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask
        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)
        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True

        gamma = K.sigmoid(K.bias_add(K.dot(inputs, self.crohis_decomp_weight), self.crohis_decomp_bias))
        crohis = cross_att + crohis_tm * (1 - gamma)
        # print('crohis:', crohis)
        output = (1 - self.tau) * output + self.tau * (K.tanh(K.bias_add(K.dot(crohis, self.crohis_weight_w),
                                                                         self.crohis_bias)))
        # h = (1 - self.tau) * h + self.tau * \
        #     (K.tanh(K.bias_add(
        #         (K.dot(crohis, self.crohis_weight_w)),
        #         self.crohis_bias)))
        #
        # print('h:', h)
        return output, [output, crohis]


class DscaRNN(RNN):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            warnings.warn('The `implementation` argument '
                          'in `DscaRNN` has been deprecated. '
                          'Please remove it from your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = DscaRNNCell(units,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           recurrent_initializer=recurrent_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           recurrent_regularizer=recurrent_regularizer,
                           bias_regularizer=bias_regularizer,
                           kernel_constraint=kernel_constraint,
                           recurrent_constraint=recurrent_constraint,
                           bias_constraint=bias_constraint,
                           dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           seqx1_dim=seqx1_dim,
                           seqx2_dim=seqx2_dim,
                           seqx1_emb_dim=seqx1_emb_dim,
                           seqx2_emb_dim=seqx2_emb_dim,
                           tau=tau)
        super(DscaRNN, self).__init__(cell,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      go_backwards=go_backwards,
                                      stateful=stateful,
                                      unroll=unroll,
                                      **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(DscaRNN, self).call(inputs,
                                         mask=mask,
                                         training=training,
                                         initial_state=initial_state)


class DscaLSTMCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        super(DscaLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.units, self.units, seqx1_emb_dim + seqx2_emb_dim)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        self.seqx1_dim = seqx1_dim
        self.seqx2_dim = seqx2_dim
        self.seqx1_emb_dim = seqx1_emb_dim
        self.seqx2_emb_dim = seqx2_emb_dim
        self.tau = tau

    def build(self, input_shape):
        margex_dim = self.seqx1_emb_dim + self.seqx2_emb_dim

        if type(self.recurrent_initializer).__name__ == 'Identity':
            def recurrent_identity(shape, gain=1., dtype=None):
                del dtype
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        self.embed_weight_1 = self.add_weight(shape=(self.seqx1_dim, self.seqx1_emb_dim),
                                              name='embed_weight_1',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.embed_weight_2 = self.add_weight(shape=(self.seqx2_dim, self.seqx2_emb_dim),
                                              name='embed_weight_2',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_w = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_u = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_v = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_w = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_u = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_v = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.crohis_weight_w = self.add_weight(shape=(margex_dim, self.units),
                                               name='crohis_weight_w',
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)

        self.crohis_bias = self.add_weight(shape=(self.units,),
                                           name='crohis_bias',
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)

        self.crohis_decomp_weight = self.add_weight(shape=(margex_dim, margex_dim),
                                                    name='crohis_decomp_weight',
                                                    initializer=self.kernel_initializer,
                                                    regularizer=self.kernel_regularizer,
                                                    constraint=self.kernel_constraint)

        self.crohis_decomp_bias = self.add_weight(shape=(margex_dim,),
                                                  name='crohis_decomp_bias',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)

        self.kernel = self.add_weight(shape=(margex_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                @K.eager
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, self.units: self.units * 2])
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, self.units * 2: self.units * 3])
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        #  print(states)
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        crohis_tm = states[2]

        seqx1 = inputs[:, :self.seqx1_dim]
        seqx2 = inputs[:, self.seqx1_dim:]

        embseqx1 = K.dot(seqx1, self.embed_weight_1)
        embseqx2 = K.dot(seqx2, self.embed_weight_2)

        # cross_att_1to2 = K.softmax(K.tanh(
        #     K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
        #     K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
        #                            K.dot(embseqx2, self.cross_attention_weight_1to2_v))
        #
        # #print('cross_att_1to2:', cross_att_1to2)
        # cross_att_2to1 = K.softmax(K.tanh(
        #     K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
        #     K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
        #                            K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        # #print('cross_att_2to1:', cross_att_2to1)

        cross_att_1to2 = K.softmax(K.tanh(
            K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
            K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
                                   K.dot(embseqx2, self.cross_attention_weight_1to2_v))

        #  print('cross_att_1to2:', cross_att_1to2)
        cross_att_2to1 = K.softmax(K.tanh(
            K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
            K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
                                   K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        #  print('cross_att_2to1:', cross_att_2to1)

        cross_att = K.concatenate([cross_att_1to2, cross_att_2to1])

        #  print('cross_att:', cross_att)
        # crohis = self.gamma * cross_att + (1 - self.gamma) * crohis_tm
        # # print('crohis:', crohis)
        # inputs = K.concatenate([embseqx1, embseqx2]) * crohis
        inputs = K.concatenate([embseqx1 * cross_att_1to2, embseqx2 * cross_att_2to1])
        #  print('inputs:', inputs)
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                      self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                      self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                            self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                      self.recurrent_kernel_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        gamma = K.sigmoid(K.bias_add(K.dot(inputs, self.crohis_decomp_weight), self.crohis_decomp_bias))
        crohis = cross_att + crohis_tm * (1 - gamma)
        # print('crohis:', crohis)
        h = (1 - self.tau) * h + self.tau * (K.tanh(K.bias_add(K.dot(crohis, self.crohis_weight_w), self.crohis_bias)))
        # h = (1 - self.tau) * h + self.tau * \
        #     (K.tanh(K.bias_add(
        #         (K.dot(crohis, self.crohis_weight_w)),
        #         self.crohis_bias)))
        #
        # print('h:', h)

        return h, [h, c, crohis]


class DscaLSTM(RNN):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = DscaLSTMCell(units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            recurrent_initializer=recurrent_initializer,
                            unit_forget_bias=unit_forget_bias,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            recurrent_regularizer=recurrent_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            recurrent_constraint=recurrent_constraint,
                            bias_constraint=bias_constraint,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            implementation=implementation,
                            seqx1_dim=seqx1_dim,
                            seqx2_dim=seqx2_dim,
                            seqx1_emb_dim=seqx1_emb_dim,
                            seqx2_emb_dim=seqx2_emb_dim,
                            tau=tau)
        super(DscaLSTM, self).__init__(cell,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       stateful=stateful,
                                       unroll=unroll,
                                       **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(DscaLSTM, self).call(inputs,
                                          mask=mask,
                                          training=training,
                                          initial_state=initial_state)


class DscaGRUCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 reset_after=False,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        super(DscaGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = (self.units, seqx1_emb_dim + seqx2_emb_dim)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        self.seqx1_dim = seqx1_dim
        self.seqx2_dim = seqx2_dim
        self.seqx1_emb_dim = seqx1_emb_dim
        self.seqx2_emb_dim = seqx2_emb_dim
        self.tau = tau

    def build(self, input_shape):

        margex_dim = self.seqx1_emb_dim + self.seqx2_emb_dim

        if isinstance(self.recurrent_initializer, initializers.Identity):
            def recurrent_identity(shape, gain=1., dtype=None):
                del dtype
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        self.embed_weight_1 = self.add_weight(shape=(self.seqx1_dim, self.seqx1_emb_dim),
                                              name='embed_weight_1',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.embed_weight_2 = self.add_weight(shape=(self.seqx2_dim, self.seqx2_emb_dim),
                                              name='embed_weight_2',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_w = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_u = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_1to2_v = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx1_emb_dim),
                                                             name='cross_attention_weight_1to2_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_w = self.add_weight(shape=(self.seqx2_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_w',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_u = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_u',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.cross_attention_weight_2to1_v = self.add_weight(shape=(self.seqx1_emb_dim, self.seqx2_emb_dim),
                                                             name='cross_attention_weight_2to1_v',
                                                             initializer=self.kernel_initializer,
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)

        self.crohis_weight_w = self.add_weight(shape=(margex_dim, self.units),
                                               name='crohis_weight_w',
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)

        self.crohis_bias = self.add_weight(shape=(self.units,),
                                           name='crohis_bias',
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)

        self.crohis_decomp_weight = self.add_weight(shape=(margex_dim, margex_dim),
                                                    name='crohis_decomp_weight',
                                                    initializer=self.kernel_initializer,
                                                    regularizer=self.kernel_regularizer,
                                                    constraint=self.kernel_constraint)

        self.crohis_decomp_bias = self.add_weight(shape=(margex_dim,),
                                                  name='crohis_decomp_bias',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)

        self.kernel = self.add_weight(shape=(margex_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3),
                                                name='recurrent_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNN DscaGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])

        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
        self.built = True

    def call(self, inputs, states, training=None):
        # print(states)
        h_tm1 = states[0]
        crohis_tm = states[1]

        seqx1 = inputs[:, :self.seqx1_dim]
        seqx2 = inputs[:, self.seqx1_dim:]

        embseqx1 = K.dot(seqx1, self.embed_weight_1)
        embseqx2 = K.dot(seqx2, self.embed_weight_2)

        # cross_att_1to2 = K.softmax(K.tanh(
        #     K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
        #     K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
        #                            K.dot(embseqx2, self.cross_attention_weight_1to2_v))
        #
        # #print('cross_att_1to2:', cross_att_1to2)
        # cross_att_2to1 = K.softmax(K.tanh(
        #     K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
        #     K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
        #                            K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        # #print('cross_att_2to1:', cross_att_2to1)

        cross_att_1to2 = K.softmax(K.tanh(
            K.dot(embseqx1, self.cross_attention_weight_1to2_w) +
            K.dot(embseqx2, self.cross_attention_weight_1to2_u)) *
                                   K.dot(embseqx2, self.cross_attention_weight_1to2_v))

        # print('cross_att_1to2:', cross_att_1to2)
        cross_att_2to1 = K.softmax(K.tanh(
            K.dot(embseqx2, self.cross_attention_weight_2to1_w) +
            K.dot(embseqx1, self.cross_attention_weight_2to1_u)) *
                                   K.dot(embseqx1, self.cross_attention_weight_2to1_v))
        # print('cross_att_2to1:', cross_att_2to1)

        cross_att = K.concatenate([cross_att_1to2, cross_att_2to1])

        # print('cross_att:', cross_att)

        inputs = K.concatenate([embseqx1 * cross_att_1to2, embseqx2 * cross_att_2to1])

        # print('inputs:', inputs)

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.input_bias_z)
                x_r = K.bias_add(x_r, self.input_bias_r)
                x_h = K.bias_add(x_h, self.input_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
                recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, self.input_bias)
            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:]

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        # print('z:', z)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        gamma = K.sigmoid(K.bias_add(K.dot(inputs, self.crohis_decomp_weight), self.crohis_decomp_bias))
        crohis = cross_att + crohis_tm * (1 - gamma)
        # print('crohis:', crohis)
        h = (1 - self.tau) * h + self.tau * (K.tanh(K.bias_add(K.dot(crohis, self.crohis_weight_w), self.crohis_bias)))
        # h = (1 - self.tau) * h + self.tau * \
        #     (K.tanh(K.bias_add(
        #         (K.dot(crohis, self.crohis_weight_w)),
        #         self.crohis_bias)))
        #
        # print('h:', h)

        return h, [h, crohis]


class DscaGRU(RNN):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 seqx1_dim=0,
                 seqx2_dim=0,
                 seqx1_emb_dim=0,
                 seqx2_emb_dim=0,
                 tau=0.5,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = DscaGRUCell(units,
                           activation=activation,
                           recurrent_activation=recurrent_activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           recurrent_initializer=recurrent_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           recurrent_regularizer=recurrent_regularizer,
                           bias_regularizer=bias_regularizer,
                           kernel_constraint=kernel_constraint,
                           recurrent_constraint=recurrent_constraint,
                           bias_constraint=bias_constraint,
                           dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           implementation=implementation,
                           reset_after=reset_after,
                           seqx1_dim=seqx1_dim,
                           seqx2_dim=seqx2_dim,
                           seqx1_emb_dim=seqx1_emb_dim,
                           seqx2_emb_dim=seqx2_emb_dim,
                           tau=tau)
        super(DscaGRU, self).__init__(cell,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      go_backwards=go_backwards,
                                      stateful=stateful,
                                      unroll=unroll,
                                      **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(DscaGRU, self).call(inputs,
                                         mask=mask,
                                         training=training,
                                         initial_state=initial_state)

