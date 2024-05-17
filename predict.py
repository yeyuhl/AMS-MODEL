import cv2
import numpy as np

from retinaface import Retinaface


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img


# 进行预测，对图片中的人脸进行识别
def predict(path):
    retinaface = Retinaface()
    image = cv_imread(path)
    if image is None:
        return 'Open Error! Try again!'
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_id = retinaface.detect_image(image)
        print(face_id)
        return face_id
