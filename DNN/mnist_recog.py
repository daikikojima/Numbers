import chainer
from chainer import serializers
from PIL import ImageOps
import chainer.links as L
from chainer import Variable
import numpy as np

from  DNN import models

def predict_num(img):
    model = L.Classifier(models.MLP(1000, 10))
    serializers.load_npz("./DNN/mymodel.npz", model)

    img_resize = ImageOps.grayscale(img.resize((28, 28)))
    data = np.array(img_resize, dtype=np.float32)
    x = data
    y = model.predictor(Variable(np.array([x]))).data.argmax(axis=1)[0]
    return y