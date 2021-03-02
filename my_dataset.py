from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOC2007DataSet(Dataset):
    '''
    下载数据，初始化数据，都可以在这里完成

    '''
    def __init__(self, voc_root, transforms=None, train_set = True):
        '''
        :param voc_root: 数据集所在的根目录(存放VOCdevkit的目录)
        :param transforms: 预处理方法
        :param train_set: 如果是True，则实例返回train.txt, 如果是False， 返回val.txt
        '''
        self.root = os.path.join(voc_root, 'VOCdevkit', 'VOC2007')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        # read train.txt or val.txt according to 'train_set'
        if train_set:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')

        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in read.readlines()] # 按行读文件
            # print(self.xml_list[0])

        # read class_dict
        try:
            json_file = open('./pascal_voc_classes.json', 'r')
            self.class_dict = json.load(json_file) # 读取json文件
        except Exception as e:
            print(e)
            exit(-1)

        # define transforms
        self.transforms = transforms

    def __len__(self):
         return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read() # 读取整个文件
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation'] # {'folder': 'VOC2007', 'filename': '000017.jpg', 'source': {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007' ...
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')

        # 保存图片的bbox, label, 是否难检测（是否和其他目标有重叠）
        boxes = []
        labels = []
        iscrowd = [] # 为0时为单目标，比较好检测

        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax]) # box = [[...], [...], [...]] （axis=0代表的是每个图片）
            labels.append(self.class_dict[obj['name']]) # 将xml文件中的'dog'等标签转化为数字标签（根据class_dict）
            iscrowd.append(int(obj['difficult']))

        # covert everything into torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx]) # 这里加[]是为了让idx的shape为1，在torch里面好计算
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0]) # 这种方式返回所有相乘的元素，自动构成一个数组

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target;

    # 复制于源码
    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        return data_height, data_width



    # 将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    def parse_xml_to_dict(self, xml):
        """
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}



# 以下为测试图片解析功能得代码


import transforms
from draw_box_utils import draw_box # 借鉴自tensorflow的代码
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
category_index = {}
try:
    json_file = open('./pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

train_data_set = VOC2007DataSet('/home/shaolun/PYTHON/object-detection/faster-rcnn.pytorch/data/', data_transform["train"], True).__getitem__(1)

# print('---------------')
#
# # load train data set
# train_data_set = VOC2007DataSet('/home/shaolun/PYTHON/object-detection/faster-rcnn.pytorch/data/', data_transform["train"], True)
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()


