from typing import Optional

import torch
from torch.nn import functional as F

EPSILON = 1e-8


class ISTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        normalized: bool = False,
    ):
        """
        Implementation of inverse Short-Time Fourier Transform (ISTFT) in PyTorch
        for ONNX export. Window sum is computed dynamically using conv_transpose1d.
        
        Parameters
        ----------
        n_fft: Size of Fourier transform
        hop_length: The distance between neighboring sliding window frames.
        win_length: The size of window frame and STFT filter. (Default: ``n_fft``)
        window: The optional window function. Shape must be 1d and `<= n_fft`. (Default: ``torch.ones(win_length)``)
        normalized: Whether the STFT was normalized. (Default: ``False``)
        """
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.normalized = normalized
        
        scale = self.n_fft / self.hop_length
        fourier_basis = torch.fft.fft(torch.eye(self.n_fft))

        cutoff = int((self.n_fft / 2 + 1))
        fourier_basis_sliced = torch.vstack(
            [torch.real(fourier_basis[:cutoff, :]), torch.imag(fourier_basis[:cutoff, :])]
        )
        inverse_basis = torch.linalg.pinv(scale * fourier_basis_sliced).transpose(0, 1).unsqueeze(1).float()
        
        fft_window = window
        if fft_window is None:
            fft_window = torch.ones(self.win_length)
        assert n_fft >= self.win_length
        fft_window = pad_center(fft_window, target_length=n_fft)
        
        # Window the bases
        inverse_basis *= fft_window
        
        # Store squared window as a conv kernel for dynamic window_sum computation
        # Shape: [1, 1, n_fft] for conv_transpose1d
        win_sq_kernel = (fft_window ** 2).view(1, 1, -1)
        
        self.register_buffer("inverse_basis", inverse_basis.float(), persistent=False)
        self.register_buffer("win_sq_kernel", win_sq_kernel.float(), persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT forward pass.

        :param input: tensor with shape [batch, freq_bins, time, 2]
                      where last dim is [real, imag]
        :return: waveform tensor [batch, samples]
        """
        assert input.shape[-1] == 2, "Last dimension must be 2 for real and imaginary parts"
        n_frames = input.shape[2]
        
        real, img = input[..., 0], input[..., 1]
        recombine_magnitude_phase = torch.cat([real, img], dim=1)
        
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )
        
        # Compute window_sum dynamically using conv_transpose1d
        # This is the same overlap-add operation but with ones instead of STFT data
        ones = torch.ones(1, 1, n_frames, device=input.device, dtype=input.dtype)
        window_sum = F.conv_transpose1d(
            ones,
            self.win_sq_kernel,
            stride=self.hop_length,
            padding=0,
        ).squeeze(0).squeeze(0)  # [output_length]
        
        # Slice to match actual output size (should already match, but be safe)
        win_dim = inverse_transform.size(-1)
        window_sum_valid = window_sum[:win_dim]
        
        # Remove modulation effects
        inverse_transform = inverse_transform / (window_sum_valid + EPSILON)
        inverse_transform = inverse_transform.squeeze(dim=1)
        inverse_transform *= float(self.n_fft) / self.hop_length

        # Trim padding from center=True STFT
        inverse_transform = inverse_transform[:, int(self.n_fft / 2):]
        inverse_transform = inverse_transform[:, :-int(self.n_fft / 2)]

        if self.normalized:
            inverse_transform = inverse_transform * torch.sqrt(torch.tensor(self.n_fft))
        return inverse_transform




def pad_center(data: torch.Tensor, target_length: int, axis: int = -1, pad_value: float = 0) -> torch.Tensor:
    """
    Center-pads a tensor along a specified axis to a target size.

    Args:
        data (torch.Tensor): The input tensor to pad.
        target_length (int): The target size along the specified axis.
        axis (int): The axis along which to pad the tensor.
        pad_value (float, optional): The value to use for padding. Defaults is 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    # Get the current size of the tensor along the specified axis
    current_len = data.shape[axis]

    # If the current size is already equal to the target size, return the original tensor
    if current_len == target_length:
        return data

    # Calculate the amount of padding needed on each side
    total_padding = target_length - current_len
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left

    # Create a padding tuple for torch.nn.functional.pad
    # torch.nn.functional.pad expects padding in reverse order of dimensions
    # and pairs for the beginning and end of each dimension
    if axis < 0:
        axis = data.dim() + axis
    pad_width = [0] * (2 * data.dim())
    pad_width[(data.dim() - 1 - axis) * 2] = pad_left
    pad_width[(data.dim() - 1 - axis) * 2 + 1] = pad_right

    # Apply padding
    padded_data = torch.nn.functional.pad(data, pad=pad_width, mode="constant", value=pad_value)

    return padded_data