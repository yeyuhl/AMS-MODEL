import threading

from flask import Flask, request, json

from encoding import encoding, recoding
from predict import predict

app = Flask(__name__)

lock = threading.Lock()


# 人脸编码
@app.route('/encode', methods=["post"])
def encode():
    return_dict = {'code': 200, 'info': '处理成功', 'result': None}
    path = request.values.get('path')
    name = request.values.get('imageName')
    # 必须上锁，不然会出现并发bug
    with lock:
        result = encoding(path, name)
    if result == "success":
        return json.dumps(return_dict, ensure_ascii=False)
    else:
        return_dict['code'] = 404
        return_dict['info'] = '请求错误'
        return json.dumps(return_dict, ensure_ascii=False)


# 人脸重编码
@app.route('/recode', methods=["post"])
def recode():
    path = request.values.get('path')
    recoding(path)
    return {'code': 200}


# 人脸识别
@app.route('/recognize', methods=["post"])
def recognize():
    return_dict = {'code': 200, 'info': '处理成功', 'result': None}
    path = request.values.get('imagePath')
    face_names = predict(path)
    if face_names == 'Open Error! Try again!':
        return_dict['code'] = 404
        return_dict['info'] = '请求错误'
        return json.dumps(return_dict, ensure_ascii=False)
    else:
        return_dict['result'] = face_names
        return json.dumps(return_dict, ensure_ascii=False)


if __name__ == '__main__':
    app.run()
