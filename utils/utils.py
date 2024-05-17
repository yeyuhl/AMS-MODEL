import math

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------#
#   对输入图像进行resize，并用灰条填充图像周围空白区域
# ---------------------------------------------------#
def letterbox_image(image, size):
    #   高度、宽度、通道数
    ih, iw, _ = np.shape(image)

    #   目标图像的宽度和高度
    w, h = size
    #   取图像宽度和高度缩放比例的最小值，以确保图像不会被拉伸或压缩
    scale = min(w / iw, h / ih)
    #   计算缩放后图像的宽度，高度
    nw = int(iw * scale)
    nh = int(ih * scale)

    #   使用OpenCV将图像缩放为指定尺寸
    image = cv2.resize(image, (nw, nh))
    #   创建一个与目标图像尺寸相同的新图像，并填充为灰色（RGB 值为 [128, 128, 128]）
    new_image = np.ones([size[1], size[0], 3]) * 128
    #   将缩放后的图像填充到新图像的中心位置
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# ---------------------------------#
#   计算人脸距离
# ---------------------------------#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    #   计算每个已知人脸特征向量和待比较人脸特征向量之间的差值
    #   axis=1指定沿着行方向计算范数，即计算每个已知人脸和待比较人脸之间的距离
    #   这里计算的距离一般指欧氏距离
    #   (n,)
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# ---------------------------------#
#   比较人脸
# ---------------------------------#
#   传入的参数是：已知的人脸特征向量，当前的人脸特征向量，门限
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    #   计算人脸特征向量之间的距离
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    #   和门限比对
    return list(dis <= tolerance), dis


# -------------------------------------#
#   人脸对齐
# -------------------------------------#
def Alignment_1(img, landmark):
    #   x是左右眼角之间的水平距离，y是其垂直距离（存在争议）
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    #   眼睛连线相对于水平线的倾斜角
    #   如果x是水平距离，那么应该只有y=0才是倾斜角为0，当然也有可能这里x和y是倒转的，即y才是水平距离
    if x == 0:
        angle = 0
    else:
        #   计算倾斜角，并转换成它的弧度制
        angle = math.atan(y / x) * 180 / math.pi
    #   获取图像中心点坐标
    center = (img.shape[1] // 2, img.shape[0] // 2)

    #   使用cv2.getRotationMatrix2D函数计算旋转矩阵，旋转矩阵用于将图像绕中心点旋转指定角度
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    #   仿射函数，进行旋转
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    #   存储旋转后的人脸关键点坐标
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark
