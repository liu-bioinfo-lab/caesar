"""
Input shapes:

    Hi-C: n_bins * n_bins

    Epigenetic marks: n_bins * n_channels

    *Each bin corresponds to a region at the size of map resolution (e.g., 200 bp)
    **Each channel corresponds to one epigenetic mark (e.g., H3K4me1 and CTCF)

Basic structure of the model:

    Input Epigenetic data -----Conv1D----> Hidden 0

    Hidden 0 -----GraphConv----> Hidden i -----GraphConv----> Hidden ii ...

    Hidden 0 -----Conv1D----> Hidden a -----Conv1D----> Hidden b ...

    [Hidden i, Hidden ii, ..., Hidden a, Hidden b, Hi-C] -----Concatenate----> Hidden X

    Hidden X -----Fully connected layer----> Out 1

    Out 1 -----Transpose----> Out 2

    [Out 1, Out 2] ----Add----> Output (Must be symmetric matrices)

"""
from keras.layers import Input, Concatenate, Permute, Conv1D, Add, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from layers import GraphConv


def model_fn(first_layer=[64, 15], gcn_layers=[96, 96, 96],
             conv_layer_filters=[64], conv_layer_windows=[15],
             nBins=1250, nMarks=9, lr=0.0001, verbose=1):
    """
    Build the GCN model

    Args:
        first_layer (list): n_filters and n_windows for the layer after input
        gcn_layers (list): n_filters for the following GCN layers
        conv_layer_filters (list): n_filters for the Conv1D layers
        conv_layer_windows (list): n_windows for the Conv1D layers
        nBins (int): size of input matrices
        nMarks (int):
        verbose (int):

    Return:
         model (keras.Model.model)
    """

    hic = Input(shape=(nBins, nBins))
    epi_data = Input(shape=(nBins + first_layer[1] - 1, nMarks))

    hidden_0 = Conv1D(first_layer[0], first_layer[1], activation='relu')(epi_data)

    if len(gcn_layers) > 0:
        hidden_g = [GraphConv(gcn_layers[0], activation='relu')([hidden_0, hic])]
        for i in range(1, len(gcn_layers)):
            hidden_g.append(GraphConv(gcn_layers[i], activation='relu')([hidden_g[-1], hic]))
    else:
        hidden_g = []

    if len(conv_layer_filters) > 0:
        hidden_c = [Conv1D(conv_layer_filters[0], conv_layer_windows[0], padding='same', activation='relu')(hidden_0)]
        for i in range(1, len(conv_layer_filters)):
            hidden_c.append(Conv1D(conv_layer_filters[i], conv_layer_windows[i],
                                   padding='same', activation='relu')(hidden_c[-1]))
    else:
        hidden_c = []

    combined = Concatenate(axis=-1)(hidden_g + hidden_c + [hic])
    pred = Conv1D(nBins, 1, activation='relu')(combined)
    pred_T = Permute([2, 1])(pred)
    res = Add()([pred, pred_T])

    m = Model(inputs=[hic, epi_data], outputs=res)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    return m


