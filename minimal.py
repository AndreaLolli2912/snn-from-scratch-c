import torch.nn as nn
import snntorch as snn
import torch
from snntorch import spikegen
num_inputs = 20
num_hidden = 21
num_outputs = 200
num_steps = 25
threshold_hidden = 0.45
threshold_output = 0.75
beta_hidden = 0.8
beta_output = 0.4
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        print("One hidden layer")
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta_hidden, threshold=threshold_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta_output, threshold= threshold_output)
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        input_spikes = spikegen.rate(x, num_steps=num_steps)
        for step in range(num_steps):
            current_input_spikes = input_spikes[step,:, :]
            cur1 = self.fc1(current_input_spikes)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)