

class Model(object):

    def __init__(self,net,loss,optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self,inputs):
        return self.net.forward(inputs)

    def backward(self,pred,target):
        loss = self.loss.loss(pred,target)
        grad = self.loss.grad(pred,target)
        grads = self.net.backward(grad)
        return loss ,grads

    def apply_grads(self, grads):
        params = self.net.params
        self.optimizer.step(grads, params)