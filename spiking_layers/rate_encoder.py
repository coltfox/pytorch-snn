import torch
from .spiking_layer import SpikingLayer


class RateEncoder(SpikingLayer):
    """Encodes an input as spikes using rate encoding"""

    def __init__(self, n_neurons: int, batch_size: int = 500, v_thr: float = 1.0, device: torch.device | str = 'cpu', gain: float = 1.0, bias: float = 0.0) -> None:
        """
        Args:
            n_neurons (int): Number of neurons in the layer
            batch_size (int): Size of batches. Default 500.
            v_thr (float): Voltage threshold. Default 1.0
            device (torch.device | str): Device to allocate tensors. Default 'cpu'
            gain (float): Neuron gain multiplier. Default 1.0
            bias (float): Neuron bias. Default 0.0
        """
        super().__init__(n_neurons, batch_size, v_thr, device)
        self.gain = gain
        self.bias = bias

    def spike(self) -> torch.Tensor:
        v_diff = self._v - self._v_thr
        spikes = torch.zeros_like(self._v)
        spikes[v_diff >= 0] = 1

        return spikes

    def encode(self, x_t: torch.Tensor) -> torch.Tensor:
        c = self.gain * x_t + self.bias
        self.calculate_voltage(c)

        return self.spike_and_reset_voltage()
