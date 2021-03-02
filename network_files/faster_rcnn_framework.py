import torch
from torch import nn
from collections import OrderedDict
# from network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
# from network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, Dict, List, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
# from network_files.transform  import GeneralizedRCNNTransform

class FasterRCNNBase(nn.Module):
    '''
        定义Faster-rcnn网络，把backbone，rpn和后续得fast-rcnn得模块串起来
        :param backbone:
        :param rpn:
        :param roi_heads: 获取feature和proposal from RPN， 计算detections from it
        :param transform: 图像预处理
    '''
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections


    # 完成FasterRCNN的前向传播
    def forward(self, images, targets = None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        '''
        :param images: (list[Tensor]): images to be processed
        :param targets: (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        :return: (dict[Tensor] or list[BoxList])
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
        '''

        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError('Expected target boxes to be a tensor'
                                         'of shape [N, 4], got {:}.'.format(
                            boxes.shape
                        ))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # 用来存储原始的图片尺寸，括号内的是对变量的格式进行一个声明
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            # 取图片的高和宽
            val = img.shape[-2:]
            assert len(val) == 2 # 防止输入的是一个一维向量
            original_image_sizes.append((val[0], val[1]))

        # 开始走pipeline==============>
        images, targets = self.transform(images, targets) # 对图像进行预处理
        features = self.backbone(images.tensors) # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor): # # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)]) # 若在多层特征层上预测，传入的就是一个有序字典




FasterRNN = FasterRCNNBase(1,1,1,1)