import torch
import torch.nn as nn
from torch.nn import Module, Linear

class LinearStateSpaceModel(Module):
    def __init__(self, num_states):
        super(LinearStateSpaceModel, self).__init__()

        self.pre_gain = nn.Parameter(torch.FloatTensor([1.0]))
        self.num_states = num_states

        self.state_and_input_to_output_layer = Linear(num_states + 1, 1, bias = False)

        self.cell = LinearStateSpaceCell(num_states)

    def forward(self, input, initial_states = None):
        batch_size = input.shape[0]
        sequence_length = input.shape[1]

        device = input.device

        input = input * self.pre_gain

        hidden = torch.zeros((batch_size, self.num_states)).to(device)
        if initial_states is not None:
            hidden[:,:] = initial_states

        predicted_output_sequence = torch.zeros(batch_size, sequence_length, 1).to(device)

        states_sequence = torch.zeros((batch_size, sequence_length, self.num_states)).to(device)

        for i in range(sequence_length - 1):
            hidden = self.cell(input[:,i, :], hidden)
            states_sequence     [:,i+1,:] = hidden[:,:]

        predicted_output_sequence = self.state_and_input_to_output_layer(torch.cat([input, states_sequence],-1))

        if initial_states is None:
            return predicted_output_sequence
        else:
            return predicted_output_sequence, states_sequence

class LinearStateSpaceCell(Module):
    def __init__(self, num_states):
        super(LinearStateSpaceCell, self).__init__()

        self.num_states = num_states

        self.state_to_state_layer = Linear(num_states, num_states, bias = False)
        self.input_to_state_layer = Linear(1, num_states, bias = False)
        bound = 1.0 / (self.state_to_state_layer.in_features)
        nn.init.uniform_(self.state_to_state_layer.weight,-bound,bound)

    def forward(self, input, in_states):
        state_output = self.state_to_state_layer(in_states) + self.input_to_state_layer(input)
        return state_output
