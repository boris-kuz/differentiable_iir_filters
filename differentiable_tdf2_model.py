from math import pi
import numpy as np
import torch
from torch.nn import Module, ModuleList, Parameter
from torch import FloatTensor
from numpy.random import uniform

class DTDFIICell(Module):
    def __init__(self):
        super(DTDFIICell, self).__init__()
        self.b0 = Parameter(FloatTensor([uniform(-1, 1)]))
        self.b1 = Parameter(FloatTensor([uniform(-1, 1)]))
        self.b2 = Parameter(FloatTensor([uniform(-1, 1)]))
        self.a1 = Parameter(FloatTensor([uniform(-0.5, 0.5)]))
        self.a2 = Parameter(FloatTensor([uniform(-0.5, 0.5)]))

    def _cat(self, vectors):
        return torch.cat([v_.unsqueeze(-1) for v_ in vectors], dim=-1)

    def forward(self, input, v):
        output = input * self.b0 + v[:, 0]
        v = self._cat([(input * self.b1 + v[:, 1] - output * self.a1), (input * self.b2 - output * self.a2)])
        return output, v

    def init_states(self, size):
        v = torch.zeros(size, 2).to(next(self.parameters()).device)
        return v

class DTDFII(Module):
    def __init__(self):
        super(DTDFII, self).__init__()
        self.cell = DTDFIICell()

    def forward(self, input, initial_states=None):
        batch_size = input.shape[0]
        sequence_length = input.shape[1]

        if initial_states is None:
            states = self.cell.init_states(batch_size)
        else:
            states = initial_states

        out_sequence = torch.zeros(input.shape[:-1]).to(input.device)
        for s_idx in range(sequence_length):
            out_sequence[:, s_idx], states = self.cell(input[:, s_idx].view(-1), states)
        out_sequence = out_sequence.unsqueeze(-1)

        if initial_states is None:
            return out_sequence
        else:
            return out_sequence, states
