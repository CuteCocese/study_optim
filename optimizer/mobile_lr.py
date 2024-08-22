def sqrt(x):
    return x**0.5



class RMSProp:
    def __init__(self, lr=1e-2, beta=0.9, eps=1e-8):
        """
        Class Parameters:
            :param lr: 학습률
            :param beta: ema(지수 이동 평균) 계수
            :param eps: 작은 상수

        Class Attributes:
            v: 학습률 조정
        """
        self.name = 'RMSProp'
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = 0.

    def update(self, x, dx):
        self.v = self.beta * self.v + (1. - self.beta) * (dx**2)
        x = x - self.lr * 1/sqrt(self.v + self.eps) * dx
        return x




