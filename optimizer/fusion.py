from ..utils import sqrt


class Adam:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Class Parameters:
            :param lr: 학습률
            :param betas: ema(지수 이동 평균) 계수
            :param eps: 작은 상수

        Class Attributes:
            m: 모멘텀
            v: 학습률 조정
        """
        self.name = 'Adam'
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = 0.
        self.v = 0.

    def update(self, x, dx):
        self.m = self.beta1 * self.m + (1. - self.beta1) * dx
        self.v = self.beta2 * self.v + (1. - self.beta2) * (dx**2)

        m_hat = self.m / (1. - self.beta1)
        v_hat = self.v / (1. - self.beta2)

        x = x - self.lr * m_hat / sqrt(v_hat + self.eps) * dx
        return x