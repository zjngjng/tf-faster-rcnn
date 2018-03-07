# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, depth, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride[0]
  shift_y = np.arange(0, height) * feat_stride[1]
  shift_z = np.arange(0, depth) * feat_stride[2]

  shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 6)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length
