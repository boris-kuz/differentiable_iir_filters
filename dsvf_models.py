from math import pi
import torch
import numpy as np
from torch.nn import Module, Parameter, ModuleList
from torch import FloatTensor
from numpy.random import uniform

class DSVFCell(Module):
    def __init__(self, G=0.5, twoR=1, hp_gain=0.0, bp_gain=0.0, lp_gain=1.0):
        args = locals()
        del args['self']
        del args['__class__']
        super(DSVFCell, self).__init__()
        for key in args:
            setattr(self, key, Parameter(FloatTensor([args[key]])))
        self.master_gain = Parameter(FloatTensor([1.0]))

    def forward(self, x, v):
        coeff0, coeff1 = self.calc_coeffs()
        input_minus_v1 = x - v[:, 1]
        bp_out = coeff1 * input_minus_v1 + coeff0 * v[:, 0]
        lp_out = self.G * bp_out + v[:, 1]
        hp_out = x - lp_out - self.twoR * bp_out
        v = torch.cat([(2 * bp_out).unsqueeze(-1), (2 * lp_out).unsqueeze(-1)], dim=-1) - v
        y = self.master_gain * (self.hp_gain * hp_out + self.bp_gain * self.twoR * bp_out + self.lp_gain * lp_out)
        return y, v

    def init_states(self, size):
        v = torch.zeros(size, 2).to(next(self.parameters()).device)
        return v

    def calc_coeffs(self):
        self.G.data = torch.clamp(self.G, min=1e-8)
        self.twoR.data = torch.clamp(self.twoR, min=0)
        self.bp_gain.data = torch.clamp(self.bp_gain, min=-1)
        self.hp_gain.data = torch.clamp(self.hp_gain, min=-1, max=1)
        self.lp_gain.data = torch.clamp(self.lp_gain, min=-1, max=1)

        coeff0 = 1.0 / (1.0 + self.G * (self.G + self.twoR))
        coeff1 = self.G * coeff0

        return coeff0, coeff1

class DSVF(Module):
    def __init__(self, G=0.5, twoR=1, hp_gain=1.0, bp_gain=1.0, lp_gain=1.0):
        super(DSVF, self).__init__()
        self.cell = DSVFCell(G=G, twoR=twoR, hp_gain=hp_gain, bp_gain=bp_gain, lp_gain=lp_gain)

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
