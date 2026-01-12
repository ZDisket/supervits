import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    #self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    self.conv1 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
    self.norm1 = nn.LayerNorm(filter_channels)
    self.act1 = nn.ReLU()
    self.drop1 = nn.Dropout(p_dropout)

    self.conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
    self.norm2 = nn.LayerNorm(filter_channels)
    self.act2 = nn.ReLU()
    
    
    # GRU for sequential modeling before projection
    self.gru = nn.GRU(filter_channels, filter_channels, num_layers=1, batch_first=True, bidirectional=False)
    self.drop2 = nn.Dropout(p_dropout * 0.5)
    
    self.is_export = False
    
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def run_rnn(self, x, x_mask):
    """Run GRU with proper packing/unpacking for variable-length sequences.
    
    Args:
        x: Input tensor of shape [batch, channels, time]
        x_mask: Mask tensor of shape [batch, 1, time]
        is_export: Whether to skip packing/unpacking for ONNX export
    
    Returns:
        Output tensor of shape [batch, channels, time]
    """
    if self.is_export:
      # Simplified flow for ONNX export: skip packing/unpacking
      x_t = x.transpose(1, 2)
      x_out, _ = self.gru(x_t)
      x_out = x_out.transpose(1, 2)
      return x_out * x_mask

    # Get lengths from mask: sum over time dimension
    x_lengths = x_mask.squeeze(1).sum(dim=1).long()  # [batch]
    
    # Transpose for RNN: [batch, channels, time] -> [batch, time, channels]
    x_t = x.transpose(1, 2)
    
    # Sort by length (required for pack_padded_sequence)
    x_lengths_sorted, sort_idx = x_lengths.sort(descending=True)
    x_sorted = x_t[sort_idx]
    
    # Clamp lengths to be at least 1 to avoid errors with empty sequences
    x_lengths_clamped = x_lengths_sorted.clamp(min=1).cpu()
    
    # Pack, run through GRU, unpack
    packed = pack_padded_sequence(x_sorted, x_lengths_clamped, batch_first=True, enforce_sorted=True)
    packed_out, _ = self.gru(packed)
    unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x_t.size(1))
    
    # Unsort to restore original order
    _, unsort_idx = sort_idx.sort()
    x_out = unpacked[unsort_idx]
    
    # Transpose back: [batch, time, channels] -> [batch, channels, time]
    x_out = x_out.transpose(1, 2)
    
    return x_out

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    #x = self.convs(x, x_mask)
    x = x * x_mask
    x = self.conv1(x) * x_mask
    x = self.act1(x)
    x = self.norm1(x.transpose(1,2)).transpose(1,2) * x_mask
    x = self.drop1(x)
    
    x = self.conv2(x) * x_mask
    x = self.act2(x)
    x = self.norm2(x.transpose(1,2)).transpose(1,2) * x_mask
    x = self.drop1(x)
    
    
    # Apply GRU for sequential modeling
    x = self.run_rnn(x, x_mask)
    x = self.drop2(x) * x_mask
    
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    
    # GRU for sequential modeling before projection
    self.gru = nn.GRU(filter_channels, filter_channels, num_layers=1, batch_first=True, bidirectional=False)
    self.drop2 = nn.Dropout(p_dropout * 0.2)
    
    self.proj = nn.Conv1d(filter_channels, 1, 1)
    
    self.is_export = False

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def run_rnn(self, x, x_mask):
    """Run GRU with proper packing/unpacking for variable-length sequences.
    
    Args:
        x: Input tensor of shape [batch, channels, time]
        x_mask: Mask tensor of shape [batch, 1, time]
        is_export: Whether to skip packing/unpacking for ONNX export
    
    Returns:
        Output tensor of shape [batch, channels, time]
    """
    if self.is_export:
      # Simplified flow for ONNX export: skip packing/unpacking
      x_t = x.transpose(1, 2)
      x_out, _ = self.gru(x_t)
      x_out = x_out.transpose(1, 2)
      return x_out * x_mask

    # Get lengths from mask: sum over time dimension
    x_lengths = x_mask.squeeze(1).sum(dim=1).long()  # [batch]
    
    # Transpose for RNN: [batch, channels, time] -> [batch, time, channels]
    x_t = x.transpose(1, 2)
    
    # Sort by length (required for pack_padded_sequence)
    x_lengths_sorted, sort_idx = x_lengths.sort(descending=True)
    x_sorted = x_t[sort_idx]
    
    # Clamp lengths to be at least 1 to avoid errors with empty sequences
    x_lengths_clamped = x_lengths_sorted.clamp(min=1).cpu()
    
    # Pack, run through GRU, unpack
    packed = pack_padded_sequence(x_sorted, x_lengths_clamped, batch_first=True, enforce_sorted=True)
    packed_out, _ = self.gru(packed)
    unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x_t.size(1))
    
    # Unsort to restore original order
    _, unsort_idx = sort_idx.sort()
    x_out = unpacked[unsort_idx]
    
    # Transpose back: [batch, time, channels] -> [batch, channels, time]
    x_out = x_out.transpose(1, 2)
    
    return x_out

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    
    # Apply GRU for sequential modeling
    x = self.run_rnn(x, x_mask)
    x = self.drop2(x)
    
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    self.emb_norm = nn.LayerNorm(hidden_channels)
    self.emb_drop = nn.Dropout(0.1)

    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      start_i_increment=2)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x)
    x = self.emb_norm(x)
    x = self.emb_drop(x)
    
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, use_transformer=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      use_transformer=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only
    self.use_transformer = use_transformer

    if self.use_transformer:
      self.attention = attentions.MultiHeadAttention(hidden_channels, hidden_channels, 2, p_dropout=p_dropout, start_i_increment=4)
    else:
      self.transformer = None

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)

    h = self.pre(x0) * x_mask

    if self.use_transformer:
      attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
      residual = h * x_mask
      residual = self.attention(residual, residual, attn_mask)
      h = h + residual

    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x




class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask



class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, gen_istft_n_fft=16, gen_istft_hop_size=4, gen_istft_win_size=16):
        super(Generator, self).__init__()
        self._is_export = False
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.gen_istft_win_size = gen_istft_win_size
        
        self.conv_post = Conv1d(ch, (gen_istft_n_fft // 2 + 1) * 2, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        self.ups_snakes = nn.ModuleList([modules.Snake1d(upsample_initial_channel // (2**i)) for i in range(len(self.ups))])
        self.post_snake = modules.Snake1d(ch)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        
        self.init_istft_params()

    @property
    def is_export(self):
        return self._is_export
    
    @is_export.setter
    def is_export(self, value):
        self._is_export = value
        if value and not hasattr(self, '_export_istft'):
            # Lazily initialize ONNX-compatible ISTFT module on first export
            from torchistft import ISTFT
            
            # Determine device from existing parameters to ensure the new module matches
            device = self.conv_pre.weight.device
            
            self._export_istft = ISTFT(
                n_fft=self.gen_istft_n_fft,
                hop_length=self.gen_istft_hop_size,
                win_length=self.gen_istft_win_size,
                window=torch.hann_window(self.gen_istft_win_size),
            ).to(device)

    def init_istft_params(self):
        self.register_buffer("istft_window", torch.hann_window(self.gen_istft_win_size))

    @staticmethod
    def safe_atan2(y, x, eps=1e-8):
        """Numerically stable atan2 that avoids zero gradients."""
        # Add small epsilon to avoid zero denominator in gradient computation
        return torch.atan2(y, x + eps * (x.abs() < eps).float())

    def forward(self, x, g=None, length=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = self.ups_snakes[i](x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = self.post_snake(x)
        x = self.conv_post(x)
        
        mag_raw = x[:, :self.gen_istft_n_fft // 2 + 1, :]
        # prevent HF ringing
        mag_raw = torch.clamp(mag_raw, max=5.5)   
        spec = torch.exp(mag_raw)
        
        phase_angle = x[:, self.gen_istft_n_fft // 2 + 1:, :] 
        real = spec * torch.cos(phase_angle)
        imag = spec * torch.sin(phase_angle)
        
        # We need to construct a complex-like tensor for torch.istft 
        # Modern pytorch requires complex input
        if self._is_export:
            # Use ONNX-compatible custom ISTFT
            # Format: [batch, freq_bins, time, 2] where last dim is [real, imag]
            stft_export = torch.stack([real, imag], dim=-1)
            x = self._export_istft(stft_export)
        else:
            stft = torch.complex(real, imag)
            x = torch.istft(
                stft, 
                n_fft=self.gen_istft_n_fft, 
                hop_length=self.gen_istft_hop_size, 
                win_length=self.gen_istft_win_size, 
                window=self.istft_window, 
                center=True,
                length=length
            )
        return x.unsqueeze(1)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    gen_istft_n_fft=16, 
    gen_istft_hop_size=4, 
    gen_istft_win_size=16,
    hop_length=None,
    **kwargs):

    super().__init__()
    self._is_export = False
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.segment_size_waveform = segment_size * hop_length
    self.n_speakers = n_speakers
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.gen_istft_n_fft = gen_istft_n_fft
    self.gen_istft_hop_size = gen_istft_hop_size
    self.gen_istft_win_size = gen_istft_win_size

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, gen_istft_n_fft=gen_istft_n_fft, gen_istft_hop_size=gen_istft_hop_size, gen_istft_win_size=gen_istft_win_size)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  @property
  def is_export(self):
    return self._is_export

  @is_export.setter
  def is_export(self, value):
    self._is_export = value
    if hasattr(self, 'dp'):
      self.dp.is_export = value
    if hasattr(self, 'dec'):
      self.dec.is_export = value

  @staticmethod
  def compute_duration_loss(predicted_logw, target_logw, mask):
    """
    Compute Mean Squared Error loss for duration prediction in log-space.
    
    This calculates how well the duration predictor matches the ground truth durations
    (extracted from monotonic alignment search). The loss is computed on log-durations
    to ensure the model learns duration ratios rather than absolute differences.
    
    Args:
        predicted_logw: Predicted log-durations from duration predictor [batch, 1, time]
        target_logw: Ground truth log-durations from attention weights [batch, 1, time]
        mask: Binary mask indicating valid (non-padded) positions [batch, 1, time]
    
    Returns:
        Scalar loss value: MSE averaged over all valid frames in the batch
    """
    # Compute squared error between predicted and target log-durations
    squared_error = (predicted_logw - target_logw) ** 2
    
    # Sum over channels (dim=1) and time (dim=2), then average by total valid frames
    return torch.sum(squared_error, [1, 2]) / torch.sum(mask)

  @staticmethod
  def compute_temporal_consistency_loss(predicted_logw, target_logw, mask, weight=0.1):
    """
    Compute temporal consistency loss for duration predictions (1st order).
    
    This penalizes differences in how predicted vs ground truth durations evolve over time.
    It compares the temporal dynamics (rate of change) between consecutive tokens, ensuring
    the model learns not just the absolute durations but also their progression patterns.
    
    Args:
        predicted_logw: Predicted log-durations from duration predictor [batch, 1, time]
        target_logw: Ground truth log-durations from attention weights [batch, 1, time]
        mask: Binary mask indicating valid (non-padded) positions [batch, 1, time]
        weight: Scaling factor for the temporal consistency loss (default: 0.1)
    
    Returns:
        Scalar loss value: MSE between predicted and target duration evolution
    """
    # Compute how durations evolve (differences between consecutive frames)
    # Predicted evolution: logw[t] - logw[t-1]
    pred_evolution = predicted_logw[:, :, 1:] - predicted_logw[:, :, :-1]
    
    # Ground truth evolution: logw_[t] - logw_[t-1]
    target_evolution = target_logw[:, :, 1:] - target_logw[:, :, :-1]
    
    # Create mask for valid consecutive pairs (both positions must be valid)
    consecutive_mask = mask[:, :, :-1] * mask[:, :, 1:]
    
    # Compute MSE between predicted and target evolution patterns
    evolution_diff = (pred_evolution - target_evolution) ** 2
    masked_diff = evolution_diff * consecutive_mask
    
    # Average over all valid consecutive pairs
    num_valid_pairs = torch.sum(consecutive_mask).clamp(min=1.0)  # Avoid division by zero
    temporal_loss = torch.sum(masked_diff) / num_valid_pairs
    
    return weight * temporal_loss

  @staticmethod
  def compute_duration_acceleration_loss(predicted_logw, target_logw, mask, weight=0.1):
    """
    Compute 2nd order duration loss (acceleration matching).
    
    This penalizes differences in how the rate of duration change evolves.
    The 2nd derivative captures "acceleration" - whether durations are speeding up
    or slowing down in their rate of change. This helps the model learn fine-grained
    temporal dynamics like gradual speed-ups before pauses or slow-downs for emphasis.
    
    2nd derivative: logw[i+1] - 2*logw[i] + logw[i-1]
    
    Args:
        predicted_logw: Predicted log-durations from duration predictor [batch, 1, time]
        target_logw: Ground truth log-durations from attention weights [batch, 1, time]
        mask: Binary mask indicating valid (non-padded) positions [batch, 1, time]
        weight: Scaling factor for the acceleration loss (default: 0.1)
    
    Returns:
        Scalar loss value: MSE between predicted and target duration acceleration
    """
    # Compute 2nd derivative (acceleration): logw[i+1] - 2*logw[i] + logw[i-1]
    # This is equivalent to: (logw[i+1] - logw[i]) - (logw[i] - logw[i-1])
    pred_accel = predicted_logw[:, :, 2:] - 2 * predicted_logw[:, :, 1:-1] + predicted_logw[:, :, :-2]
    target_accel = target_logw[:, :, 2:] - 2 * target_logw[:, :, 1:-1] + target_logw[:, :, :-2]
    
    # Create mask for valid triplets (all three positions must be valid)
    triplet_mask = mask[:, :, :-2] * mask[:, :, 1:-1] * mask[:, :, 2:]
    
    # Compute MSE between predicted and target acceleration patterns
    accel_diff = (pred_accel - target_accel) ** 2
    masked_diff = accel_diff * triplet_mask
    
    # Average over all valid triplets
    num_valid_triplets = torch.sum(triplet_mask).clamp(min=1.0)  # Avoid division by zero
    accel_loss = torch.sum(masked_diff) / num_valid_triplets
    
    return weight * accel_loss

  def forward(self, x, x_lengths, y, y_lengths, sid=None):

    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      import monotonic_align
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log1p(w) * x_mask  # Ground truth log1p-durations from alignment
      logw = self.dp(x, x_mask, g=g)  # Predicted log1p-durations
      
      # Compute primary duration loss (MSE between predicted and target)
      l_length = self.compute_duration_loss(logw, logw_, x_mask)
      
      # Add 1st order loss: match velocity (rate of change) patterns
      l_temporal = self.compute_temporal_consistency_loss(logw, logw_, x_mask, weight=0.1)  # ~14x larger
      
      # Add 2nd order loss: match acceleration patterns
      l_accel = self.compute_duration_acceleration_loss(logw, logw_, x_mask, weight=0.02)  # ~43x larger
      
      l_length = l_length + l_temporal + l_accel 

    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    # Calculate expected output length: segment_size is the waveform length in training
    output_length = self.segment_size_waveform
    o = self.dec(z_slice, g=g, length=output_length)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)
    w = torch.expm1(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

  def voice_enhancement(self, y, y_lengths, sid=None, noise_scale=0.0, bypass_flow=False):
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)
    else:
      g = None
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    
    if noise_scale > 0:
      z_input = (m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale) * y_mask
    else:
      z_input = m_q * y_mask
      
    if bypass_flow:
      z_hat = z_input
    else:
      z_p = self.flow(z_input, y_mask, g=g)
      z_hat = self.flow(z_p, y_mask, g=g, reverse=True)
      
    o_hat = self.dec(z_hat * y_mask, g=g)
    return o_hat, y_mask, (z, m_q, z_hat)

  def resynthesis(self, x, x_lengths, y_ref, y_ref_lengths, sid=None, noise_scale=0.667):
    """
    Re-synthesize speech by extracting durations from reference audio
    and applying them to text input.
    
    Args:
        x: Text input tensor [batch, text_length]
        x_lengths: Text lengths [batch]
        y_ref: Reference audio spectrogram [batch, channels, time]
        y_ref_lengths: Reference audio lengths [batch]
        sid: Speaker ID (optional for multi-speaker)
        noise_scale: Sampling noise scale (0.0 for deterministic)
    
    Returns:
        o: Output waveform
        attn: Extracted alignment/durations
        y_mask: Output mask
    """
    # Encode text
    x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    
    # Get speaker embedding
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)
    else:
      g = None
    
    # Encode reference audio to latent space
    z, m_q, logs_q, y_mask = self.enc_q(y_ref, y_ref_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)
    
    # Compute alignment between reference audio and text
    # Same logic as training forward pass
    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_x]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_x]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_y, t_x]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_y, t_x]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_x]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
      
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
    
    # Expand priors using extracted alignment
    m_p_expanded = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p_expanded = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    
    # Sample from expanded priors
    if noise_scale > 0:
      z_p_new = m_p_expanded + torch.randn_like(m_p_expanded) * torch.exp(logs_p_expanded) * noise_scale
    else:
      z_p_new = m_p_expanded
    
    # Decode through flow and generator
    z_out = self.flow(z_p_new, y_mask, g=g, reverse=True)
    o = self.dec(z_out * y_mask, g=g)
    
    return o, attn, y_mask

