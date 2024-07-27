import torch
from typing import Iterable
from spiking_layers.dense_spiking_layer import DenseSpikingLayer
from spiking_layers.spiking_layer import SpikingLayer


class DenseSNN(torch.nn.Module):
    """Spiking Neural Network consisting of only dense layers"""

    def __init__(self, n_ts: int, input_encoder: SpikingLayer, hidden_layers: Iterable[DenseSpikingLayer], output_layer: DenseSpikingLayer):
        """
        Args:
            n_ts (int): Number of timesteps
            input_encoder (SpikingLayer): Layer to encode inputs as spikes
            dense_layers (Iterable[DenseSpikingLayer]): Iterable of dense spiking layers
            output_layer (DenseSpikingLayer): The dense output layer
        """
        super().__init__()
        self.n_ts = n_ts
        self.encoder_layer = input_encoder
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.output_layer = output_layer

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (Tensor): Network input
        """
        self._reset_all_layers()

        return self._forward_through_time(x)

    def _reset_all_layers(self):
        """Re-initialize all network layers"""
        self.encoder_layer.reset_layer()

        for layer in self.hidden_layers:
            layer.reset_layer()

        self.output_layer.reset_layer()

    def _forward_through_time(self, x: torch.Tensor):
        """
        Propogate x forward through time

        Args:
            x (Tensor): Network input
        """
        batch_size = self.encoder_layer._v.shape[0]
        num_classes = self.output_layer._v.shape[1]
        ts_spikes = torch.zeros(batch_size, self.n_ts, num_classes)

        for t in range(0, self.n_ts):
            spikes = self.encoder_layer.encode(x)
            for hidden_layer in self.hidden_layers:
                spikes = hidden_layer(spikes)
            spikes = self.output_layer(spikes)

            ts_spikes[:, t, :] = spikes

        return ts_spikes
