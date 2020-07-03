"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from typing import Tuple, Dict

import torch
import torch.nn as nn

from codebase.modelzoo.head import get_head
from codebase.modelzoo.basemodel import BaseModel


class EALSTM(BaseModel):
    def __init__(self, cfg: Dict):
        super(EALSTM, self).__init__(cfg=cfg)

        # parse config into class attributes
        self.input_size_dyn = len(cfg["dynamic_inputs"])
        self.input_size_stat = len(cfg["static_inputs"] + cfg["camels_attributes"])
        if cfg["use_basin_id_encoding"]:
            self.input_size_stat += cfg["number_of_basins"]
        self.hidden_size = cfg["hidden_size"]
        self.initial_forget_bias = cfg["initial_forget_bias"]

        self.input_net = nn.Linear(self.input_size_stat, self.hidden_size)

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(self.input_size_dyn, 3 * self.hidden_size)
        )
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(self.hidden_size, 3 * self.hidden_size)
        )
        self.bias = nn.Parameter(torch.FloatTensor(3 * self.hidden_size))

        self.dropout = nn.Dropout(p=cfg["output_dropout"])

        self.head = get_head(cfg=cfg, n_in=cfg["hidden_size"], n_out=self.output_size)

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[: self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, x_one_hot: torch.Tensor):

        if (x_s.nelement() > 0) and (x_one_hot.nelement() > 0):
            x_s = torch.cat([x_s, x_one_hot], dim=-1)
        elif x_one_hot.nelement() > 0:
            x_s = x_one_hot
        else:
            pass

        x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())

        # calculate input gate only once because inputs are static
        i = torch.sigmoid(self.input_net(x_s))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(
                x_d[t], self.weight_ih
            )
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        # not batch_first -> h_n.transpose(0, 1)
        y_hat = self.head(self.dropout(h_n))

        return y_hat, h_n, c_n
