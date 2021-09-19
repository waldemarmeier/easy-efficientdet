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
"""Bounding Box List operations for Numpy BoxLists.
Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import, division, print_function

import np_box_list
import np_box_ops
import numpy as np


class SortOrder(object):
    """Enum class for sort order.
  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
    ASCEND = 1
    DESCEND = 2


def iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
    return np_box_ops.iou(boxlist1.get(), boxlist2.get())


def ioa(boxlist1, boxlist2):
    """Computes pairwise intersection-over-area between box collections.
  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
    return np_box_ops.ioa(boxlist1.get(), boxlist2.get())


def sort_by_field(boxlist, field, order=SortOrder.DESCEND):
    """Sort boxes and associated fields according to a scalar field.
  A common use case is reordering the boxes according to descending scores.
  Args:
    boxlist: BoxList holding N boxes.
    field: A BoxList field for sorting and reordering the BoxList.
    order: (Optional) 'descend' or 'ascend'. Default is descend.
  Returns:
    sorted_boxlist: A sorted BoxList with the field in the specified order.
  Raises:
    ValueError: if specified field does not exist or is not of single dimension.
    ValueError: if the order is not either descend or ascend.
  """
    if not boxlist.has_field(field):
        raise ValueError('Field ' + field + ' does not exist')
    if len(boxlist.get_field(field).shape) != 1:
        raise ValueError('Field ' + field + 'should be single dimension.')
    if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
        raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    sorted_indices = np.argsort(field_to_sort)
    if order == SortOrder.DESCEND:
        sorted_indices = sorted_indices[::-1]
    return gather(boxlist, sorted_indices)


def gather(boxlist, indices, fields=None):
    """Gather boxes from BoxList according to indices and return new BoxList.
  By default, gather returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.
  Args:
    boxlist: BoxList holding N boxes
    indices: a 1-d numpy array of type int_
    fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int_
  """
    if indices.size:
        if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
            raise ValueError('indices are out of valid range.')
    subboxlist = np_box_list.BoxList(boxlist.get()[indices, :])
    if fields is None:
        fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        subboxlist.add_field(field, extra_field_data[indices, ...])
    return subboxlist


def filter_scores_greater_than(boxlist, thresh):
    """Filter to keep only boxes with score exceeding a given threshold.
  This op keeps the collection of boxes whose corresponding scores are
  greater than the input threshold.
  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.
    thresh: scalar threshold
  Returns:
    a BoxList holding M boxes where M <= N
  Raises:
    ValueError: if boxlist not a BoxList object or if it does not
      have a scores field
  """
    if not isinstance(boxlist, np_box_list.BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')
    scores = boxlist.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape '
                         'consistent with [None, 1]')
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
                                    [-1]).astype(np.int32)
    return gather(boxlist, high_score_indices)


def non_max_suppression(boxlist,
                        max_output_size=10000,
                        iou_threshold=1.0,
                        score_threshold=-10.0):
    """Non maximum suppression.
  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes. In each iteration, the detected bounding box with
  highest score in the available pool is selected.
  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores. All scores belong to the same class.
    max_output_size: maximum number of retained boxes
    iou_threshold: intersection over union threshold.
    score_threshold: minimum score threshold. Remove the boxes with scores
                     less than this value. Default value is set to -10. A very
                     low threshold to pass pretty much all the boxes, unless
                     the user sets a different score threshold.
  Returns:
    a BoxList holding M boxes where M <= max_output_size
  Raises:
    ValueError: if 'scores' field does not exist
    ValueError: if threshold is not in [0, 1]
    ValueError: if max_output_size < 0
  """
    if not boxlist.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    boxlist = filter_scores_greater_than(boxlist, score_threshold)
    if boxlist.num_boxes() == 0:
        return boxlist

    boxlist = sort_by_field(boxlist, 'scores')

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if boxlist.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(boxlist, selected_indices)
        else:
            return boxlist

    boxes = boxlist.get()
    num_boxes = boxlist.num_boxes()
    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_boxes, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_boxes):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_box_ops.iou(
                    np.expand_dims(boxes[i, :], axis=0), boxes[valid_indices, :])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold)
    return gather(boxlist, np.array(selected_indices))
