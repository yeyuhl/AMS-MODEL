import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


# --------------------------------------#
#   写中文需要转成PIL来写。
# --------------------------------------#
def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


# --------------------------------------#
#   一定注意backbone和model_path的对应。
#   在更换facenet_model后，
#   一定要注意重新编码人脸。
# --------------------------------------#
class Retinaface(object):
    _defaults = {
        # ----------------------------------------------------------------------#
        #   retinaface训练完的权值路径
        #   使用WiderFace数据集训练，训练集有12880张图片，验证集有3226张图片
        # ----------------------------------------------------------------------#
        "retinaface_model_path": 'model_data/Retinaface_mobilenet0.25.pth',
        # ----------------------------------------------------------------------#
        #   retinaface所使用的主干网络，有mobilenet和resnet50
        # ----------------------------------------------------------------------#
        "retinaface_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   retinaface中只有得分大于置信度的预测框会被保留下来
        # ----------------------------------------------------------------------#
        "confidence": 0.5,
        # ----------------------------------------------------------------------#
        #   retinaface中非极大抑制所用到的nms_iou大小，nms_iou值越低，NMS过程越严格，抑制的重叠框越多
        # ----------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        # ----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        # ----------------------------------------------------------------------#
        "letterbox_image": True,

        # ----------------------------------------------------------------------#
        #   facenet训练完的权值路径
        #   使用CASIA-WebFace数据集训练，训练集有452961张图片，验证集是lfw，有13234张图片
        # ----------------------------------------------------------------------#
        "facenet_model_path": 'model_data/facenet_mobilenet.pth',
        # ----------------------------------------------------------------------#
        #   facenet所使用的主干网络， mobilenet和inception_resnetv1
        # ----------------------------------------------------------------------#
        "facenet_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   facenet所使用到的输入图片大小
        # ----------------------------------------------------------------------#
        "facenet_input_shape": [160, 160, 3],
        # ----------------------------------------------------------------------#
        #   facenet所使用的人脸距离门限，用于判断两张人脸是否相似的阈值，facenet_threhold的值越低，人脸识别过程越严格，只有两张人脸的距离非常接近时才会被认为是相似的人脸
        # ----------------------------------------------------------------------#
        "facenet_threhold": 0.9,

        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinaface
    # ---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   不同主干网络的config信息
        # ---------------------------------------------------#
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # ---------------------------------------------------#
        #   先验框的生成
        # ---------------------------------------------------#
        self.anchors = Anchors(self.cfg, image_size=(
            self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.known_face_encodings = np.load(
                "model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet = Facenet(backbone=self.facenet_backbone, mode="predict").eval()
        device = torch.device('cuda' if self.cuda else 'cpu')

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path, map_location=device)
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path, map_location=device)
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def process_face_image(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   打开人脸图片
            # ---------------------------------------------------#
            image = np.array(Image.open(path), np.float32)
            # ---------------------------------------------------#
            #   对输入图像进行一个备份
            # ---------------------------------------------------#
            old_image = image.copy()
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            #   将人脸框的相对坐标转换成绝对坐标，以便在原图上绘制人脸框和关键点，为了实现这一转换，需要计算一个比例因子即scale
            #   人脸框的相对坐标是相对于输入图像尺寸的比例
            #   例如(0.2, 0.3, 0.5, 0.7)表示人脸框的左上角坐标为输入图像宽度的20%和高度的30%，右下角坐标为输入图像宽度的50%和高度的70%
            #   将相对坐标乘以比例因子就能获得绝对坐标
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            #   将人脸关键点的相对坐标转换为绝对坐标
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            #   判断是否要对图片大小进行限制，然后进行预测框的计算
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            # ---------------------------------------------------#
            #   将处理完的图片传入Retinaface网络当中进行预测
            # ---------------------------------------------------#
            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化，即将0~255转化为0~1
                # -----------------------------------------------------------#
                #   torch.from_numpy(image)将numpy格式的图像数据转换为pytorch格式的Tensor数据
                #   preprocess_input(image)将图像数据减去一个固定的均值，可以有效地对图像数据进行归一化，提高模型的训练效果
                #   transpose(2, 0, 1)将图像数据的形状从[高度, 宽度, 通道]调整为[通道, 高度, 宽度]，这样才能传入模型
                #   unsqueeze(0) 在图像数据的第一个维度增加一个维度，使其成为一个batch，pytorch模型通常以batch的形式进行训练和预测
                #   type(torch.FloatTensor) 将数据类型转换为torch.FloatTensor
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    image = image.cuda()
                    anchors = anchors.cuda()

                #   预测框调整参数，预测框置信度，人脸关键点的调整参数
                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸检测结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                #   非极大抑制，筛选出一定区域内属于同一种类得分最大的框
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    print(names[index], "：未检测到人脸")
                    continue
                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))
            #   缩放以适应原始图像
            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#
            #   最佳人脸框位置和最大人脸框面积
            best_face_location = None
            biggest_area = 0
            #   遍历人脸检测结果中的每个预测框
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                #   预测框宽度和高度
                w = right - left
                h = bottom - top
                #    如果当前预测框的面积大于最大人脸框面积，则更新最大人脸框面积和最佳人脸框位置
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # ---------------------------------------------------#
            #   截取图像
            # ---------------------------------------------------#
            #   从原始图像中截取包含人脸的图像区域，并将其存储在 crop_img 变量中
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            #   将人脸关键点的位置调整为相对于人脸框左上角坐标的相对位置（本来是相对整张图片的左上角）
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            #   进行人脸对齐，即将歪着的人脸掰正，一般需要用到眼睛连线相对于水平线的倾斜角和图片中心这两个参数
            #   通过这两个参数对图片进行旋转
            crop_img, _ = Alignment_1(crop_img, landmark)

            #   将截取的人脸图像调整为模型输入的尺寸，并将其转换为浮点数类型，并除以 255 进行归一化
            #   将图像调整为指定尺寸，并使用灰条填充图像周围的空白区域
            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            #   通道顺序调整
            crop_img = crop_img.transpose(2, 0, 1)
            #   增加一个维度，将图像转换为 (批次大小，通道数，高度，宽度) 的格式
            crop_img = np.expand_dims(crop_img, 0)

            # ---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            # ---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()
                #   使用人脸识别模型 self.facenet 计算人脸图像的特征向量
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)
        return face_encodings

    def encode_face_dataset(self, image_paths, names):
        face_encodings = self.process_face_image(image_paths, names)
        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), names)

    def encode_single_face(self, image_paths, names):
        encodings_path = "model_data/{}_face_encoding.npy".format(self.facenet_backbone)
        names_path = "model_data/{}_names.npy".format(self.facenet_backbone)
        if os.path.exists(encodings_path) and os.path.exists(names_path):
            origin_face_encodings = np.load(encodings_path)
            origin_face_names = np.load(names_path)
            face_encodings = self.process_face_image(image_paths, names)
            new_face_encodings = np.vstack([origin_face_encodings, face_encodings])
            new_face_names = np.concatenate((origin_face_names, names), axis=0)
            np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone),
                    new_face_encodings)
            np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), new_face_names)
        else:
            self.encode_face_dataset(image_paths, names)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            # ---------------------------------------------------#
            #   如果没有预测框则返回原图
            # ---------------------------------------------------#
            if len(boxes_conf_landms) <= 0:
                return old_image

            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # ---------------------------------------------------#
        #   Retinaface检测部分-结束
        # ---------------------------------------------------#

        # -----------------------------------------------#
        #   Facenet编码部分-开始
        # -----------------------------------------------#
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            # ----------------------#
            #   人脸编码
            # ----------------------#
            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                # -----------------------------------------------#
                #   利用facenet_model计算长度为128特征向量
                # -----------------------------------------------#
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)
        # -----------------------------------------------#
        #   Facenet编码部分-结束
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   人脸特征比对-开始
        # -----------------------------------------------#
        face_names = []
        for face_encoding in face_encodings:
            # -----------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # -----------------------------------------------------#
            #   返回匹配人脸，以及二者之间的欧氏距离
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                    tolerance=self.facenet_threhold)
            name = "Unknown"
            # -----------------------------------------------------#
            #   取出这个最近人脸的评分
            #   取出当前输入进来的人脸，最接近的已知人脸的序号
            # -----------------------------------------------------#
            #   找到距离列表中最小值的索引，即最接近预测框中的人脸的已知人脸的索引
            best_match_index = np.argmin(face_distances)
            #   再找到其对应的名字
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#

        #   将识别结果绘制到原图片上
        #   遍历所有预测框
        for i, b in enumerate(boxes_conf_landms):
            #   格式化人脸框的置信度并转换为字符串
            text = "{:.4f}".format(b[4])
            #   将人脸框坐标转换为整型列表
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            #   在原图上绘制人脸框
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            #   计算人脸框中心点的横坐标
            cx = b[0]
            #   计算人脸框中心点的纵坐标，并向下偏移12个像素
            cy = b[1] + 12
            #   在人脸框上方绘制置信度，字体大小为 0.5，颜色为白色
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   绘制人脸关键点，b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return face_names
