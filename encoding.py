import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface(1)


# 对图片进行编码
def encoding(image_path, name):
    # 确保路径和名称不为空
    if image_path and name:
        # 将图片路径添加到列表中
        image_paths = [image_path]
        # 将名称添加到列表中，这里假设名称已经是正确的格式
        names = [name]
        # 调用 Retinaface 的 encode_face_dataset 方法来编码单张照片
        retinaface.encode_single_face(image_paths, names)
        return "success"
    else:
        return "failure"


# 对所有图片重新编码
def recoding(path):
    list_dir = os.listdir(path)
    image_paths = []
    names = []
    for name in list_dir:
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            image_paths.append(path + name)
            names.append(name.split(".")[0])
    retinaface.encode_face_dataset(image_paths, names)
