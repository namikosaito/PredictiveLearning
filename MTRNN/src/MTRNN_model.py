import numpy as np

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

class MTRNN(nn.Module):
    def __init__(
        self,
        layer_size={"in": 1, "out": 1, "io": 3, "cf": 4, "cs": 5},
        tau={"tau_io": 2, "tau_cf": 5.0, "tau_cs": 70.0},
        open_rate={"feature": 1, "angle": 1, "sensor": 1},
        torch_device="cpu",
    ):
        super(MTRNN, self).__init__()
        self.layer_size = layer_size
        self.tau = tau
        self.open_rate = open_rate
        self.torch_device = torch_device
        # self.last_output = None
        self.last_feature = None
        self.last_angle = None
        self.last_sensor = None
        self.i2io = nn.Linear(self.layer_size["in"], self.layer_size["io"])
        self.io2o = nn.Linear(self.layer_size["io"], self.layer_size["out"])
        self.io2io = nn.Linear(self.layer_size["io"], self.layer_size["io"])
        self.io2cf = nn.Linear(self.layer_size["io"], self.layer_size["cf"])
        self.cf2io = nn.Linear(self.layer_size["cf"], self.layer_size["io"])
        self.cf2cs = nn.Linear(self.layer_size["cf"], self.layer_size["cs"])
        self.cf2cf = nn.Linear(self.layer_size["cf"], self.layer_size["cf"])
        self.cs2cf = nn.Linear(self.layer_size["cs"], self.layer_size["cf"])
        self.cs2cs = nn.Linear(self.layer_size["cs"], self.layer_size["cs"])
        self.activate = torch.nn.Tanh()

    def init_state(self, batch_num):
        # del self.last_output, self.io_state, self.cf_state, self.cs_state
        self.last_output = None
        self.io_state = torch.zeros(size=(batch_num, self.layer_size["io"])).to(self.torch_device)
        self.cf_state = torch.zeros(size=(batch_num, self.layer_size["cf"])).to(self.torch_device)
        self.cs_state = torch.zeros(size=(batch_num, self.layer_size["cs"])).to(self.torch_device)

        # fill_value = 0
        # self.last_output = torch.full(
        #     size=(batch_size, self.layer_size["out"]), fill_value=fill_value,
        # )
        # self.io_state = torch.full(
        #     size=(batch_size, self.layer_size["io"]), fill_value=fill_value
        # )
        # self.cf_state = torch.full(
        #     size=(batch_size, self.layer_size["cf"]), fill_value=fill_value
        # )
        # self.cs_state = torch.full(
        #     size=(batch_size, self.layer_size["cs"]), fill_value=fill_value
        # )

    def _next_state(self, previous, new, tau):
        connected = torch.stack(new)
        new_summed = connected.sum(dim=0)
        ret = (1.0 - 1.0 / tau) * previous + new_summed / tau

        del connected, new_summed
        return self.activate(ret)

    def forward(self, x, test=False):  # x.shape(batch,x)
        closed_x = x
        closed_x = closed_x.view(-1, self.layer_size["in"])

        new_io_state = self._next_state(
            previous=self.io_state,
            new=[
                self.io2io(self.io_state),
                self.cf2io(self.cf_state),
                self.i2io(closed_x),
            ],
            tau=self.tau["tau_io"],
        )
        new_cf_state = self._next_state(
            previous=self.cf_state,
            new=[
                self.cf2cf(self.cf_state),
                self.cs2cf(self.cs_state),
                self.io2cf(self.io_state),
            ],
            tau=self.tau["tau_cf"],
        )
        new_cs_state = self._next_state(
            previous=self.cs_state,
            new=[
                self.cs2cs(self.cs_state),
                self.cf2cs(self.cf_state),
            ],
            tau=self.tau["tau_cs"],
        )
        self.io_state = new_io_state
        self.cf_state = new_cf_state
        self.cs_state = new_cs_state
        y = self.activate(self.io2o(self.io_state))
        self.last_output = y

        del closed_x, new_io_state, new_cf_state, new_cs_state
        if test:
            return y, self.cs_state
        else:
            return y