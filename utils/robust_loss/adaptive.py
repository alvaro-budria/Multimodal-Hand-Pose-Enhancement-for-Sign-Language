# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Implements the adaptive form of the loss.

You should only use this function if 1) you want the loss to change it's shape
during training (otherwise use general.py) or 2) you want to impose the loss on
a wavelet or DCT image representation, a only this function has easy support for
that.
"""

import numpy as np
import torch
import torch.nn as nn

from robust_loss import distribution  # import distribution
from robust_loss import util  # import util


class AdaptiveLossFunction(nn.Module):
  """The adaptive loss function on a matrix.

  This class behaves differently from general.lossfun() and
  distribution.nllfun(), which are "stateless", allow the caller to specify the
  shape and scale of the loss, and allow for arbitrary sized inputs. This
  class only allows for rank-2 inputs for the residual `x`, and expects that
  `x` is of the form [batch_index, dimension_index]. This class then
  constructs free parameters (torch Parameters) that define the alpha and scale
  parameters for each dimension of `x`, such that all alphas are in
  (`alpha_lo`, `alpha_hi`) and all scales are in (`scale_lo`, Infinity).
  The assumption is that `x` is, say, a matrix where x[i,j] corresponds to a
  pixel at location j for image i, with the idea being that all pixels at
  location j should be modeled with the same shape and scale parameters across
  all images in the batch. If the user wants to fix alpha or scale to be a
  constant,
  this can be done by setting alpha_lo=alpha_hi or scale_lo=scale_init
  respectively.
  """

  def __init__(self,
               num_dims,
               float_dtype,
               device,
               alpha_lo=1,
               alpha_hi=4,
               alpha_init=2,
               scale_lo=1e-5,
               scale_init=0.5):
    """Sets up the loss function.

    Args:
      num_dims: The number of dimensions of the input to come.
      float_dtype: The floating point precision of the inputs to come.
      device: The device to run on (cpu, cuda, etc).
      alpha_lo: The lowest possible value for loss's alpha parameters, must be
        >= 0 and a scalar. Should probably be in (0, 2).
      alpha_hi: The highest possible value for loss's alpha parameters, must be
        >= alpha_lo and a scalar. Should probably be in (0, 2).
      alpha_init: The value that the loss's alpha parameters will be initialized
        to, must be in (`alpha_lo`, `alpha_hi`), unless `alpha_lo` == `alpha_hi`
        in which case this will be ignored. Defaults to (`alpha_lo` +
        `alpha_hi`) / 2
      scale_lo: The lowest possible value for the loss's scale parameters. Must
        be > 0 and a scalar. This value may have more of an effect than you
        think, as the loss is unbounded as scale approaches zero (say, at a
        delta function).
      scale_init: The initial value used for the loss's scale parameters. This
        also defines the zero-point of the latent representation of scales, so
        SGD may cause optimization to gravitate towards producing scales near
        this value.
    """
    super(AdaptiveLossFunction, self).__init__()

    if not np.isscalar(alpha_lo):
      raise ValueError('`alpha_lo` must be a scalar, but is of type {}'.format(
          type(alpha_lo)))
    if not np.isscalar(alpha_hi):
      raise ValueError('`alpha_hi` must be a scalar, but is of type {}'.format(
          type(alpha_hi)))
    if alpha_init is not None and not np.isscalar(alpha_init):
      raise ValueError(
          '`alpha_init` must be None or a scalar, but is of type {}'.format(
              type(alpha_init)))
    if not alpha_lo >= 0:
      raise ValueError('`alpha_lo` must be >= 0, but is {}'.format(alpha_lo))
    if not alpha_hi >= alpha_lo:
      raise ValueError('`alpha_hi` = {} must be >= `alpha_lo` = {}'.format(
          alpha_hi, alpha_lo))
    if alpha_init is not None and alpha_lo != alpha_hi:
      if not (alpha_init > alpha_lo and alpha_init < alpha_hi):
        raise ValueError(
            '`alpha_init` = {} must be in (`alpha_lo`, `alpha_hi`) = ({} {})'
            .format(alpha_init, alpha_lo, alpha_hi))
    if not np.isscalar(scale_lo):
      raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(
          type(scale_lo)))
    if not np.isscalar(scale_init):
      raise ValueError(
          '`scale_init` must be a scalar, but is of type {}'.format(
              type(scale_init)))
    if not scale_lo > 0:
      raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
    if not scale_init >= scale_lo:
      raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(
          scale_init, scale_lo))

    self.num_dims = num_dims
    if float_dtype == np.float32:
      float_dtype = torch.float32
    if float_dtype == np.float64:
      float_dtype = torch.float64
    self.float_dtype = float_dtype
    self.device = device
    if isinstance(device, int) or\
       (isinstance(device, str) and 'cuda' in device) or\
       (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.set_device(self.device)

    self.distribution = distribution.Distribution()

    if alpha_lo == alpha_hi:
      # If the range of alphas is a single item, then we just fix `alpha` to be
      # a constant.
      self.fixed_alpha = torch.tensor(
          alpha_lo, dtype=self.float_dtype,
          device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
      self.alpha = lambda: self.fixed_alpha
    else:
      # Otherwise we construct a "latent" alpha variable and define `alpha`
      # As an affine function of a sigmoid on that latent variable, initialized
      # such that `alpha` starts off as `alpha_init`.
      if alpha_init is None:
        alpha_init = (alpha_lo + alpha_hi) / 2.
      latent_alpha_init = util.inv_affine_sigmoid(
          alpha_init, lo=alpha_lo, hi=alpha_hi)
      self.register_parameter(
          'latent_alpha',
          torch.nn.Parameter(
              latent_alpha_init.clone().detach().to(
                  dtype=self.float_dtype,
                  device=self.device)[np.newaxis, np.newaxis].repeat(
                      1, self.num_dims),
              requires_grad=True))
      self.alpha = lambda: util.affine_sigmoid(
          self.latent_alpha, lo=alpha_lo, hi=alpha_hi)

    if scale_lo == scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      self.fixed_scale = torch.tensor(
          scale_init, dtype=self.float_dtype,
          device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
      self.scale = lambda: self.fixed_scale
    else:
      # Otherwise we construct a "latent" scale variable and define `scale`
      # As an affine function of a softplus on that latent variable.
      self.register_parameter(
          'latent_scale',
          torch.nn.Parameter(
              torch.zeros((1, self.num_dims)).to(
                  dtype=self.float_dtype, device=self.device),
              requires_grad=True))
      self.scale = lambda: util.affine_softplus(
          self.latent_scale, lo=scale_lo, ref=scale_init)

  def lossfun(self, x, **kwargs):
    """Computes the loss on a matrix.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension must be equal to self.num_dims. Must be a tensor or
        numpy array of type self.float_dtype.
      **kwargs: Arguments to be passed to the underlying distribution.nllfun().

    Returns:
      A tensor of the same type and shape as input `x`, containing the loss at
      each element of `x`. These "losses" are actually negative log-likelihoods
      (as produced by distribution.nllfun()) and so they are not actually
      bounded from below by zero. You'll probably want to minimize their sum or
      mean.
    """
    x = torch.as_tensor(x)
    assert len(x.shape) == 2
    assert x.shape[1] == self.num_dims
    assert x.dtype == self.float_dtype
    return self.distribution.nllfun(x, self.alpha(), self.scale(), **kwargs)
