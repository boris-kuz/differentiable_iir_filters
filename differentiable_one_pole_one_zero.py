import torch
from torch.nn import Module, Parameter
from torch import FloatTensor

class DOnePoleCell(Module):
    def __init__(self, a1=0.5, b0=1.0, b1=0.0):
        super(DOnePoleCell, self).__init__()
        self.b0 = Parameter(FloatTensor([b0]))
        self.b1 = Parameter(FloatTensor([b1]))
        self.a1 = Parameter(FloatTensor([a1]))

    def init_states(self, size):
        state = torch.zeros(size).to(self.a1.device)
        return state

    def forward(self, input, state):
        self.a1.data = self.a1.clamp(-1, 1)
        output = self.b0 * input + state
        state = self.b1 * input + self.a1 * output
        return output, state

class DOnePole(Module):
    def __init__(self):
        super(DOnePole, self).__init__()
        self.cell = DOnePoleCell()

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

