import chainer
from chainer import serializers
from PIL import ImageOps, Image
import chainer.links as L
from chainer import Variable
import numpy as np

from DNN import models


def mu(z, p):
    return np.sum(z * p)


def mid(gray):
    N = np.sum([gray == i for i in range(256)])
    p = np.zeros(256)
    z = np.zeros(256)
    for i in range(256):
        z[i] = i
        p[i] = (np.sum(gray == i) / N)
    t = mu(z, p)
    rst = (gray > 128) * 255
    return rst

def squarelize(img):
    max_len = max(img.size[0], img.size[1])
    rst = Image.new("L", [max_len, max_len], (0))
    rst.paste(img, (int((max_len - img.size[0]) / 2), int((max_len - img.size[0]) / 2)))
    return rst
def predict_num(img):
    model = L.Classifier(models.MLP(1000, 10))
    serializers.load_npz("./DNN/mymodel.npz", model)
    modified = squarelize(ImageOps.grayscale(img))
    img_resize = modified.resize((28, 28))
    data = 255 - np.array(img_resize, dtype=np.float32)
    x = np.array(mid(data), dtype=np.float32)
    x = x / 255.0
    y = model.predictor(Variable(np.array([x]))).data.argmax(axis=1)[0]
    return y
