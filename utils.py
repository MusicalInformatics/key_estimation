from abc import ABC, abstractmethod
import logging
import os

from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from tqdm import tqdm
import glob
import partitura as pt
import warnings

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


def load_data(min_seq_length: int = 10) -> List[np.ndarray]:
    # load data
    files = glob.glob(os.path.join("data", "*.mid"))
    files.sort()
    sequences = []
    for fn in files:
        seq = pt.load_performance_midi(fn)[0]
        if len(seq.notes) > min_seq_length:
            sequences.append(seq.note_array())
    return sequences

    """ """

    def __init__(
        self,
        input_size,
        output_size,
        recurrent_size,
        hidden_size,
        n_layers=1,
        dropout=0.0,
        batch_first=True,
        dtype=torch.float32,
        device=None,
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.rnn = nn.GRU(
            input_size,
            self.recurrent_size,
            self.n_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )
        dense_in_features = self.recurrent_size
        self.dense = nn.Linear(
            in_features=dense_in_features,
            out_features=self.hidden_size,
        )
        self.output = nn.Linear(
            in_features=self.hidden_size,
            out_features=output_size,
        )

    def init_hidden(self, batch_size):
        n_layers = self.n_layers
        return torch.zeros(n_layers, batch_size, self.recurrent_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size).type(x.type())
        output, h = self.rnn(x, h0)
        flatten_shape = self.recurrent_size
        dense = self.dense(output.contiguous().view(-1, flatten_shape))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y

def prob_x_given_context(rnn, x, context, pitch_idxs, dur_idxs):
    # Assume that batch_size == 1, i.e., there is only one
    # sequence

    if isinstance(context, np.ndarray):
        if context.ndim == 2:
            context = context[np.newaxis, :, :]
        context = torch.tensor(context).to(rnn.dtype)
    softmax = nn.Softmax(dim=0)
    y = rnn(context)[-1, 1]
    pitch_prob = softmax(y[pitch_idxs]).detach().cpu().numpy()
    duration_prob = softmax(y[dur_idxs]).detach().cpu().numpy()

    x_pitch = x[pitch_idxs]
    x_dur = x[dur_idxs]
    pp = np.prod((pitch_prob) ** x_pitch * (1 - pitch_prob) ** (1 - x_pitch))
    dp = np.prod((duration_prob) ** x_dur * (1 - duration_prob) ** (1 - x_dur))
    p_x_given_contex = pp * dp
    return p_x_given_contex


def find_nearest(array, value):
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"), 0, len(array) - 1)
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx
