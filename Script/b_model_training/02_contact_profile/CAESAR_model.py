import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Input, Conv1D, Flatten, Dot, RepeatVector, Concatenate, Permute, Add, Multiply
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.optimizers import Adam


class GraphConv(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        input_dim = features_shape[2]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1]
        # tf.print(basis)

        norm_basis = basis / (tf.reduce_sum(basis, axis=2, keepdims=True) + tf.constant(1e-9))
        new_features = tf.einsum('bij,jk->bik', features, self.kernel)

        if self.use_bias:
            new_features += self.bias

        output = tf.einsum('bij,bjk->bik', norm_basis, new_features)
        # output = tf.matmul(norm_basis, new_features)
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def CAESAR(nBins=1250, nMarks=6, verbose=1, lr=0.0001, positional_dim=8,
           n_GC_layers=2, GC_dim=96, GC_trainable=True,
           n_Conv_layers=1, Conv_dim=96, Conv_size=15, Conv_trainable=True,
           FC_layer_dims=[], FC_trainable=True
           ):
    # Inputs
    hic = Input(shape=(nBins, nBins), name='Inp_HiC')
    epi_inp = Input(shape=(nBins, nMarks), name='Inp_epi')
    positional = Input(shape=(nBins, positional_dim), name='Inp_pos')

    epi_data = Concatenate(axis=-1, name='conc')([epi_inp, positional])

    # GC Layers
    gc_layers = [
        Conv1D(filters=GC_dim, kernel_size=15, padding='same', name=f'GC_0',
               activation='relu', trainable=GC_trainable)(epi_data)
    ]
    for i in range(n_GC_layers):
        gc_layers.append(
            GraphConv(units=GC_dim, use_bias=True, name=f'GC_{i + 1}',
                      activation='relu', trainable=GC_trainable)([gc_layers[-1], hic])
        )
    gc_outputs = Concatenate(axis=-1, name=f'GC_end')(gc_layers)

    # Conv1D layers
    conv_layers = [
        Conv1D(filters=Conv_dim, kernel_size=15, padding='same', name=f'Conv_0',
               activation='relu', trainable=Conv_trainable)(epi_data)
    ]
    for i in range(n_Conv_layers):
        conv_layers.append(
            Conv1D(filters=Conv_dim, kernel_size=Conv_size, padding='same', name=f'Conv_{i + 1}',
                   activation='relu', trainable=Conv_trainable)(conv_layers[-1])
        )
    conv_outputs = Concatenate(axis=-1, name=f'Conv_end')(conv_layers)

    # FC Layers
    fc_layers = [Concatenate(axis=-1, name=f'FC_0')([gc_outputs, conv_outputs, hic])]
    FC_layer_dims.append(nBins)
    for i, dim in enumerate(FC_layer_dims):
        fc_layers.append(
            Dense(dim, name=f'FC_{i + 1}', activation='relu', trainable=FC_trainable,)(fc_layers[-1])
        )
    fc_trans = Permute(dims=[2, 1], name=f'FC_trans')(fc_layers[-1])
    outputs = Add(name='FC_end')([fc_layers[-1], fc_trans])

    # outputs = Add(name='final')([inner_outputs, fc_outputs])

    m = Model(inputs=[hic, epi_inp, positional], outputs=outputs)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    return m


def CAESAR_loop(nBins=1250, nMarks=6, verbose=1, lr=0.0001, positional_dim=8,
           n_GC_layers=2, GC_dim=96, GC_trainable=True,
           n_Conv_layers=2, Conv_dim=96, Conv_size=5, Conv_trainable=True,
           Inner_layer_dims=[512], Inner_trainable=True,
           ):
    # Inputs
    hic = Input(shape=(nBins, nBins), name='Inp_HiC')
    epi_inp = Input(shape=(nBins, nMarks), name='Inp_epi')
    positional = Input(shape=(nBins, positional_dim), name='Inp_pos')
    mask = Input(shape=(nBins, nBins), name='Mask')

    epi_data = Concatenate(axis=-1, name='conc')([epi_inp, positional])

    # GC Layers
    gc_layers = [
        Conv1D(filters=GC_dim, kernel_size=15, padding='same', name=f'GC_0',
               activation='relu', trainable=GC_trainable)(epi_data)
    ]
    for i in range(n_GC_layers):
        gc_layers.append(
            GraphConv(units=GC_dim, use_bias=True, name=f'GC_{i + 1}',
                      activation='relu', trainable=GC_trainable)([gc_layers[-1], hic])
        )
    gc_outputs = Concatenate(axis=-1, name=f'GC_end')(gc_layers)

    # Conv1D layers
    conv_layers = [
        Conv1D(filters=Conv_dim, kernel_size=15, padding='same', name=f'Conv_0',
               activation='relu', trainable=Conv_trainable)(epi_data)
    ]
    for i in range(n_Conv_layers):
        conv_layers.append(
            Conv1D(filters=Conv_dim, kernel_size=Conv_size, padding='same', name=f'Conv_{i + 1}',
                   activation='relu', trainable=Conv_trainable)(conv_layers[-1])
        )
    conv_outputs = Concatenate(axis=-1, name=f'Conv_end')(conv_layers)

    # inner layers
    inner_layers = [Concatenate(axis=-1, name=f'Inner_0')([gc_outputs, conv_outputs])]
    for i, dim in enumerate(Inner_layer_dims):
        inner_layers.append(
            Dense(dim, name=f'Inner_{i + 1}', trainable=Inner_trainable)(inner_layers[-1])
        )
    inner_ = Dot(axes=(2, 2), name=f'Inner_end')([inner_layers[-1], inner_layers[-1]])
    outputs = Multiply(name='Inner_Mask')([inner_, mask])

    m = Model(inputs=[hic, epi_inp, positional, mask], outputs=outputs)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    return m


if __name__ == '__main__':
    CAESAR()

