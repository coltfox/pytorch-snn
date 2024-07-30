import torch
from .spiking_layer import SpikingLayer
from torch_funcs.fast_sigmoid import FastSigmoidSpike
import numpy as np


class DenseSpikingLayer(SpikingLayer, torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, batch_size: int = 500, v_thr: float = 1.0, device: torch.device | str = 'cpu', time_const: float = 1e-3, dt: float = 1e-3) -> None:
        """
        Args:
            n_in (int): Number of input neurons
            n_out (int): Number of output neurons
            batch_size (int): Size of batches. Default 500
            v_thr (float): Voltage threshold. Default 1.0
            device (torch.device | str): Device to allocate tensors. Default 'cpu'
            time_const (float): RC time constant. Default 1e-3
            dt (float): Delta t to determine current decay constant. Default 1e-3
        """
        super().__init__(n_out, batch_size, v_thr, device)

        self._c = torch.zeros(batch_size, n_out, device=device)

        self._fc = torch.nn.Linear(n_in, n_out, bias=False)
        self._fc.weight.data = torch.empty(
            n_out, n_in, device=device).normal_(mean=0, std=2 / np.sqrt(n_in))

        self._decay = torch.as_tensor(np.exp(-dt / time_const), device=device)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        self.calculate_current(x_t)
        self.calculate_voltage(self._c)

        return self.spike_and_reset_voltage()

    def spike(self) -> torch.Tensor:
        return FastSigmoidSpike.apply(self._v - self._v_thr)

    def calculate_current(self, s_t: torch.Tensor) -> torch.Tensor:
        self._c = self._decay * self._c + self._fc(s_t)

    def reset_layer(self):
        super().reset_layer()
        self._c = torch.zeros_like(self._c)
