import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, 200)
            self.l3 = L.Linear(None, 500)
            self.l4 = L.Linear(None, n_out)

    def __call__(self, x):
        h4 = F.relu(self.l1(x))
        h5 = F.relu(self.l2(h4))
        h6 = F.relu(self.l3(h5))
        return self.l4(h6)
