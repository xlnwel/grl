from tensorflow.keras import layers

from nn.cnn import cnn
from nn.mlp import *
from nn.rnns.lstm import MLSTM
from nn.rnns.gru import MGRU
from nn.dnc.dnc import DNC


def create_encoder(config, name='encoder'):
    config = config.copy()
    if 'cnn_name' in config:
        return cnn(**config, name=name)
    else:
        assert 'units_list' in config
        return mlp(**config, name=name)

Encoder = create_encoder

def mlp(units_list=[], out_size=None, **kwargs):
    return MLP(units_list, out_size=out_size, **kwargs)

def rnn(config, name):
    config = config.copy()
    rnn_name = config.pop('rnn_name')
    if rnn_name == 'gru':
        return layers.GRU(**config, name=name)
    elif rnn_name == 'mgru':
        return MGRU(config, name=name)
    elif rnn_name == 'lstm':
        return layers.LSTM(**config, name=name)
    elif rnn_name == 'mlstm':
        return MLSTM(config, name=name)
    else:
        raise ValueError(f'Unkown rnn: {rnn_name}')

def dnc_rnn(output_size, 
            access_config=dict(memory_size=128, word_size=16, num_reads=4, num_writes=1), 
            controller_config=dict(hidden_size=128),
            clip_value=20,
            name='dnc',
            rnn_config={}):
    """Return an RNN that encapsulates DNC
    
    Args:
        output_size: Output dimension size of dnc
        access_config: A dictionary of access module configuration. 
            memory_size: The number of memory slots
            word_size: The size of each memory slot
            num_reads: The number of read heads
            num_writes: The number of write heads
            name: name of the access module, optionally
        controller_config: A dictionary of controller(LSTM) module configuration
        clip_value: Clips controller and core output value to between
            `[-clip_value, clip_value]` if specified
        name: module name
        rnn_config: specifies extra arguments for keras.layers.RNN
    """
    dnc_cell = DNC(access_config, 
                controller_config, 
                output_size, 
                clip_value, 
                name)
    return layers.RNN(dnc_cell, **rnn_config)
