
# 경사 하강법
class GD:
    def __init__(self,
                 lr: float=1e-2
                 ):
        """
        :param lr: 학습률
        :return None: 아무것도 리턴 안함
        """
        self.name = 'GD'
        self.lr = lr

    def update(self, x: float, dx: float) -> float:
        x = x - self.lr * dx
        return x


# 모멘텀(관성)
class Momentum:
    def __init__(self,
                 lr: float=1e-2,
                 r: float=0.9
                 ):
        """
        :param lr: 학습률
        :param r: 관성률
        """
        self.name = 'Momentum'
        self.lr = lr
        self.r = r
        self.momentum = 0.

    def update(self,
               x: float,
               dx: float
               ) -> float:
        self.momentum = self.r * self.momentum + self.lr * dx
        x = x - self.momentum
        return x
