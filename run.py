import numpy as np
import matplotlib.pyplot as plt
from optimizer import GD, Momentum, RMSProp, Adam


# Optimizers 옵티마이저(최적화기)
optimizer = [
    GD(lr=1e-3),
    Momentum(lr=1e-3, r=0.9),
    RMSProp(lr=1e-3, beta=0.9, eps=1e-8),
    Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8),
]
