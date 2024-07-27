from abc import abstractmethod, ABC
import torch


class SpikingLayer(ABC):
    """Spiking layer for a Spiking Neural Network"""

    def __init__(self, n_neurons: int, batch_size: int = 500, v_thr: float = 1.0, device: torch.device | str = 'cpu') -> None:
        """
        Args:
            n_neurons (int): Number of neurons in the layer
            batch_size (int): Size of batches. Default 500
            v_thr (float): Voltage threshold. Default 1.0
            device (torch.device | str): Device to allocate tensors. Default 'cpu'
        """
        super().__init__()
        self._v = torch.zeros(batch_size, n_neurons, device=device)
        self._v_thr = torch.as_tensor(v_thr, device=device)

    def calculate_voltage(self, c: torch.Tensor):
        """
        Calculate voltage given a current

        Args:
            c: Current
        """
        self._v = self._v + c
        self._v[self._v < 0] = 0.

    def reset_layer(self):
        """Reset voltages to 0"""
        self._v = torch.zeros_like(self._v)

    def reset_spiked_neurons(self, spikes: torch.Tensor):
        """Reset the voltage of spiked neurons"""
        self._v[spikes > 0] = 0.

    def spike_and_reset_voltage(self):
        """
        Spike neurons with sufficient voltage and reset the spiked neurons

        Returns:
            spikes (Tensor): Tensor of shape (batch_size, n_neurons) consisting of 1's and 0's 
        """
        spikes = self.spike()
        self.reset_spiked_neurons(spikes)

        return spikes

    @abstractmethod
    def spike(self) -> torch.Tensor:
        """
        Get spikes based on current membrane potential

        Returns:
            spikes (Tensor): Tensor of shape (batch_size, n_neurons) consisting of 1's and 0's 
        """
        ...
