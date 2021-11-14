# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Operations for np_box_mask_list.BoxMaskList.
Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from six.moves import range

from easy_efficientdet._third_party.tf_object_detection_api.utils import (
    np_box_list_ops,
    np_box_mask_list,
    np_mask_ops,
)


def iou(box_mask_list1, box_mask_list2):
    """Computes pairwise intersection-over-union between box and mask collections.
  Args:
    box_mask_list1: BoxMaskList holding N boxes and masks
    box_mask_list2: BoxMaskList holding M boxes and masks
  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
    return np_mask_ops.iou(box_mask_list1.get_masks(), box_mask_list2.get_masks())


def ioa(box_mask_list1, box_mask_list2):
    """Computes pairwise intersection-over-area between box and mask collections.
  Intersection-over-area (ioa) between two masks mask1 and mask2 is defined as
  their intersection area over mask2's area. Note that ioa is not symmetric,
  that is, IOA(mask1, mask2) != IOA(mask2, mask1).
  Args:
    box_mask_list1: np_box_mask_list.BoxMaskList holding N boxes and masks
    box_mask_list2: np_box_mask_list.BoxMaskList holding M boxes and masks
  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
    return np_mask_ops.ioa(box_mask_list1.get_masks(), box_mask_list2.get_masks())


def box_list_to_box_mask_list(boxlist):
    """Converts a BoxList containing 'masks' into a BoxMaskList.
  Args:
    boxlist: An np_box_list.BoxList object.
  Returns:
    An np_box_mask_list.BoxMaskList object.
  Raises:
    ValueError: If boxlist does not contain `masks` as a field.
  """
    if not boxlist.has_field('masks'):
        raise ValueError('boxlist does not contain mask field.')
    box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxlist.get(),
                                                 mask_data=boxlist.get_field('masks'))
    extra_fields = boxlist.get_extra_fields()
    for key in extra_fields:
        if key != 'masks':
            box_mask_list.data[key] = boxlist.get_field(key)
    return box_mask_list


def gather(box_mask_list, indices, fields=None):
    """Gather boxes from np_box_mask_list.BoxMaskList according to indices.
  By default, gather returns boxes corresponding to the input index list, as
  well as all additional fields stored in the box_mask_list (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.
  Args:
    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes
    indices: a 1-d numpy array of type int_
    fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
  Returns:
    subbox_mask_list: a np_box_mask_list.BoxMaskList corresponding to the subset
        of the input box_mask_list specified by indices
  Raises:
    ValueError: if specified field is not contained in box_mask_list or if the
        indices are not of type int_
  """
    if fields is not None:
        if 'masks' not in fields:
            fields.append('masks')
    return box_list_to_box_mask_list(
        np_box_list_ops.gather(boxlist=box_mask_list, indices=indices, fields=fields))


def sort_by_field(box_mask_list, field, order=np_box_list_ops.SortOrder.DESCEND):
    """Sort boxes and associated fields according to a scalar field.
  A common use case is reordering the boxes according to descending scores.
  Args:
    box_mask_list: BoxMaskList holding N boxes.
    field: A BoxMaskList field for sorting and reordering the BoxMaskList.
    order: (Optional) 'descend' or 'ascend'. Default is descend.
  Returns:
    sorted_box_mask_list: A sorted BoxMaskList with the field in the specified
      order.
  """
    return box_list_to_box_mask_list(
        np_box_list_ops.sort_by_field(boxlist=box_mask_list, field=field, order=order))


def filter_scores_greater_than(box_mask_list, thresh):
    """Filter to keep only boxes and masks with score exceeding a given threshold.
  This op keeps the collection of boxes and masks whose corresponding scores are
  greater than the input threshold.
  Args:
    box_mask_list: BoxMaskList holding N boxes and masks.  Must contain a
      'scores' field representing detection scores.
    thresh: scalar threshold
  Returns:
    a BoxMaskList holding M boxes and masks where M <= N
  Raises:
    ValueError: if box_mask_list not a np_box_mask_list.BoxMaskList object or
      if it does not have a scores field
  """
    if not isinstance(box_mask_list, np_box_mask_list.BoxMaskList):
        raise ValueError('box_mask_list must be a BoxMaskList')
    if not box_mask_list.has_field('scores'):
        raise ValueError('input box_mask_list must have \'scores\' field')
    scores = box_mask_list.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape '
                         'consistent with [None, 1]')
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
                                    [-1]).astype(np.int32)
    return gather(box_mask_list, high_score_indices)


def non_max_suppression(box_mask_list,
                        max_output_size=10000,
                        iou_threshold=1.0,
                        score_threshold=-10.0):
    """Non maximum suppression.
  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes. In each iteration, the detected bounding box with
  highest score in the available pool is selected.
  Args:
    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes.  Must contain
      a 'scores' field representing detection scores. All scores belong to the
      same class.
    max_output_size: maximum number of retained boxes
    iou_threshold: intersection over union threshold.
    score_threshold: minimum score threshold. Remove the boxes with scores
                     less than this value. Default value is set to -10. A very
                     low threshold to pass pretty much all the boxes, unless
                     the user sets a different score threshold.
  Returns:
    an np_box_mask_list.BoxMaskList holding M boxes where M <= max_output_size
  Raises:
    ValueError: if 'scores' field does not exist
    ValueError: if threshold is not in [0, 1]
    ValueError: if max_output_size < 0
  """
    if not box_mask_list.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    box_mask_list = filter_scores_greater_than(box_mask_list, score_threshold)
    if box_mask_list.num_boxes() == 0:
        return box_mask_list

    box_mask_list = sort_by_field(box_mask_list, 'scores')

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if box_mask_list.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(box_mask_list, selected_indices)
        else:
            return box_mask_list

    masks = box_mask_list.get_masks()
    num_masks = box_mask_list.num_boxes()

    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_masks, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_masks):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_mask_ops.iou(np.expand_dims(masks[i], axis=0),
                                                       masks[valid_indices])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold)
    return gather(box_mask_list, np.array(selected_indices))
