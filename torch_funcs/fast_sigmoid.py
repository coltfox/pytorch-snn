import torch


class FastSigmoidSpike(torch.autograd.Function):
    """
    Spiking forward pass and backward pass using a surrogate-derivative (fast sigmoid partial derivative)
    """
    scale = 5

    @staticmethod
    def fast_sigmoid_partial(x: torch.Tensor):
        """Partial derivative of fast sigmoid. Approximation of heaviside derivative."""
        return 1 / (1 + torch.abs(x) * FastSigmoidSpike.scale)**2

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        Spike where x >= 0

        Args:
            ctx: Autograd context
            x (Tensor): Input to Heaviside step function (voltage - voltage_threshold)
        """
        ctx.save_for_backward(x)
        spikes = torch.zeros_like(x)
        spikes[x >= 0.0] = 1.0

        return spikes

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Compute gradient using fast sigmoid approximation

        Args:
            ctx: Autograd context
            grad_out (Tensor): Previous layer gradients
        """
        x: torch.Tensor = ctx.saved_tensors[0]
        return grad_out * FastSigmoidSpike.fast_sigmoid_partial(x)
