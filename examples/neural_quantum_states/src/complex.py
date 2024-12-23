import numpy as np
import torch
from typing import Union

'''
A collection of simple functions for performing complex number arithmetic on batches of doubled-up real tensors
'''
# Simplify the functions by using torch's built-in indexing for the last dimension
def real(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return x[..., 0]

def imag(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return x[..., 1]

def conjugate(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return torch.stack([real(x), -imag(x)], dim=-1)

def norm_square(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return real(x)**2 + imag(x)**2

def exp(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    amp, phase = real(x).exp(), imag(x)
    return torch.stack([amp * phase.cos(), amp * phase.sin()], dim=-1)

# Simplify the code by making sure that y is a torch tensor before converting it to x's type
def scalar_mult(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    y = y.to(x.device)
    re = real(x) * real(y) - imag(x) * imag(y)
    im = real(x) * imag(y) + imag(x) * real(y)
    return torch.stack([re, im], dim=-1)
    #return torch.stack([re, im], dim=-1) if torch.is_tensor(x) else np.stack([re, im], axis=-1)
