from faster_rcnn.config import cfg
import numpy as np
from rpn.generate_anchors import generate_anchors

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4, 8, 16, 32]):
    _anchors = generate_anchors(scale=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]


