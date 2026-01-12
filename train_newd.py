import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import SynthesizerTrn
from discriminators import Discriminator, DiscriminatorLoss, feature_loss
from audio_loss import AudioLoss
from losses import kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from text import text_to_sequence


def get_text_for_eval(text, hps):
    """Convert text to tensor for evaluation inference."""
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# Default test sentences for evaluation
DEFAULT_EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a test of the text to speech system.",
    "How are you doing today? I hope everything is going well.",
    "The weather is quite nice outside, perfect for a walk.",
    "Machine learning models can generate surprisingly natural sounding speech.",
]


_eval_text_tensors = None


torch.backends.cudnn.benchmark = False
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '55555'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      hop_length=hps.data.hop_length,
      **hps.model).cuda(rank)
  # Discriminator config with defaults for backward compatibility
  disc_cfg = getattr(hps, 'discriminator', None)
  if disc_cfg is None:
    disc_cfg = type('obj', (object,), {
        'loss_type': 'hinge',
        'use_se_blocks': False,
        'enable_mpd': True,
        'enable_msd': False,
        'enable_mbsd': True,
        'instance_noise_std': 0.0,
        'mbsd_window_lengths': [2048, 1024, 512],
        'mbsd_hop_factor': 0.25,
        'c_fm': 1.0,
    })()
  
  net_d = Discriminator(
      sample_rate=hps.data.sampling_rate,
      use_se_blocks=getattr(disc_cfg, 'use_se_blocks', False),
      enable_mpd=getattr(disc_cfg, 'enable_mpd', True),
      enable_msd=getattr(disc_cfg, 'enable_msd', False),
      enable_mbsd=getattr(disc_cfg, 'enable_mbsd', True),
      instance_noise_std=getattr(disc_cfg, 'instance_noise_std', 0.0),
      mbsd_window_lengths=getattr(disc_cfg, 'mbsd_window_lengths', [2048, 1024, 512]),
      mbsd_hop_factor=getattr(disc_cfg, 'mbsd_hop_factor', 0.25),
  ).cuda(rank)
  loss_fn = DiscriminatorLoss(loss_type=getattr(disc_cfg, 'loss_type', 'hinge'))
  c_fm = getattr(disc_cfg, 'c_fm', 1.0)
  
  # Audio loss config - convert HParams to dict for AudioLoss
  audio_loss_cfg = getattr(hps, 'audio_loss', None)
  if audio_loss_cfg is not None:
    audio_loss_dict = {k: getattr(audio_loss_cfg, k) for k in dir(audio_loss_cfg) if not k.startswith('_')}
  else:
    audio_loss_dict = {'sampling_rate': hps.data.sampling_rate, 'use_multi_scale_mel_loss': False}
  audio_loss_dict['fp16_run'] = hps.train.fp16_run  # Pass fp16 flag for eps handling
  audio_loss_fn = AudioLoss(audio_loss_dict, device=f'cuda:{rank}')
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  if getattr(hps, "pretrained", None):
    if rank == 0:
      logger.info("Loading pretrained checkpoints from %s", hps.pretrained)
    utils.load_checkpoint(utils.latest_checkpoint_path(hps.pretrained, "G_*.pth"), net_g, optimizer=None)
    utils.load_checkpoint(utils.latest_checkpoint_path(hps.pretrained, "D_*.pth"), net_d, optimizer=None)
    epoch_str = 1
    global_step = 0
  else:
    try:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      global_step = (epoch_str - 1) * len(train_loader)
    except:
      epoch_str = 1
      global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler('cuda', enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], loss_fn, c_fm, audio_loss_fn)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, loss_fn, c_fm, audio_loss_fn)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, loss_fn, c_fm, audio_loss_fn):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  
  # Gradient accumulation setup
  grad_accum_steps = getattr(hps.train, 'gradient_accumulation_steps', 1)
  
  loader = train_loader
  if rank == 0:
    loader = tqdm(train_loader, desc=f"Epoch {epoch}")

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast(device_type='cuda', enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(device_type='cuda', enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = loss_fn.discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    
    # Gradient accumulation for discriminator
    is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
    
    if not is_accumulating or batch_idx == 0:
      optim_d.zero_grad()
    
    scaler.scale(loss_disc_all / grad_accum_steps).backward()
    
    if not is_accumulating:
      scaler.unscale_(optim_d)
      grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
      scaler.step(optim_d)
    else:
      grad_norm_d = 0.0

    with autocast(device_type='cuda', enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(device_type='cuda', enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        # Perceptual audio losses (multi-scale mel, MR-STFT, pitch)
        loss_audio, audio_loss_dict = audio_loss_fn(y, y_hat)

        loss_fm = feature_loss(fmap_r, fmap_g) * c_fm
        loss_gen, losses_gen = loss_fn.generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_audio + loss_dur + loss_kl
    
    # Gradient accumulation for generator
    if not is_accumulating or batch_idx == 0:
      optim_g.zero_grad()
    
    scaler.scale(loss_gen_all / grad_accum_steps).backward()
    
    if not is_accumulating:
      scaler.unscale_(optim_g)
      grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
      scaler.step(optim_g)
      scaler.update()
    else:
      grad_norm_g = 0.0

    if rank == 0:
      loader.set_postfix(loss_g=loss_gen_all.item(), loss_d=loss_disc_all.item())

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_audio, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/audio": loss_audio, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
        
        # Log individual audio loss components
        for k, v in audio_loss_dict.items():
          scalar_dict[f"loss/g/audio/{k}"] = v

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    
    # Infer test sentences (up to N, configurable via hps.train.max_eval_texts, default 5)
    global _eval_text_tensors
    if _eval_text_tensors is None:
        eval_texts = getattr(hps.data, 'eval_texts', None) or DEFAULT_EVAL_TEXTS
        max_eval_texts = getattr(hps.train, 'max_eval_texts', 5)
        _eval_text_tensors = []
        for test_text in eval_texts[:max_eval_texts]:
            t_text_norm = get_text_for_eval(test_text, hps)
            _eval_text_tensors.append(t_text_norm)
    
    for t_idx, t_text_norm in enumerate(_eval_text_tensors):
        t_text_norm_cuda = t_text_norm.unsqueeze(0).cuda(0)
        t_text_lengths = torch.LongTensor([t_text_norm.size(0)]).cuda(0)
        
        with torch.no_grad():
          audio_test, _, t_mask, *_ = generator.module.infer(
              t_text_norm_cuda, t_text_lengths, 
              noise_scale=.667, noise_scale_w=0.8, length_scale=1.0
          )
    
          test_audio_lengths = t_mask.sum([1,2]).long() * hps.data.hop_length
          y_test_mel = mel_spectrogram_torch(
              audio_test.squeeze(1).float(),
              hps.data.filter_length,
              hps.data.n_mel_channels,
              hps.data.sampling_rate,
              hps.data.hop_length,
              hps.data.win_length,
              hps.data.mel_fmin,
              hps.data.mel_fmax
          )
          image_dict[f"gen/mel_test{t_idx}"] = utils.plot_spectrogram_to_numpy(y_test_mel[0].cpu().numpy())
          audio_dict[f"gen/audio_test{t_idx}"] = audio_test[0,:,:test_audio_lengths[0]]
    
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
