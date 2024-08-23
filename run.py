import numpy as np
import matplotlib.pyplot as plt
from optimizer import GD, Momentum, RMSProp, Adam
from utils import Logger
import sys


# 함수 설정
def target_func(x: float) -> float:
    return 10 * x ** 2 + np.sin(10 * x)


# 타겟 함수의 도함수
def grad_func(x: float) -> float:
    return 20 * x + 10 * np.sin(10 * x)


# 수치 미분
def num_diff1(func: target_func, x: float, h=1-8) -> float:
    return (func(x+h) - func(x)) / h


# 중앙 차분
def num_diff2(func: target_func, x: float, h=1e-8) -> float:
    return (func(x+h) - func(x-h)) / (2*h)


# Optimizers 옵티마이저(최적화기)
optims: list = [
    GD(lr=1e-2),
    Momentum(lr=1e-2, r=0.9),
    RMSProp(lr=1e-2, beta=0.9, eps=1e-8),
    Adam(lr=1e-2, betas=(0.9, 0.999), eps=1e-8),
]

# HYPER PARAMETER
epochs: int = 1000  # 반복 횟수
bounds: list = [-2, 2]  # 경계
x: float = np.random.uniform(*bounds)  # 초기 위치 설정(랜덤)
# x: float = 1.  # 수동

# 기타
epoch_list: list = list(range(0, epochs+1))
graph_step: float = 1e-2  # 그래프 그리는 정확도

# Set Mode
show_x_graph: bool = True  # x의 위치 변화 그래프
show_y_graph: bool = False  # y의 위치 변화 그래프
show_step_graph: bool = False  # x와 y의 위치 변화를 함수에 그림
show_optim_info: bool = False  # 옵티마이저 정보
save_info: bool = True  # 옵티마이저 정보 저장
save_graph: bool = True  # 그래프 저장 유무
save_type: str = 'png'  # pdf, jpeg 등등

# 중요치 않은 정보 필터링
info_filter: list = ['momentum', 'm', 'v', 't']

# x 그래프 초기화
if show_x_graph:
    plt.figure("X History")
    plt.title('X History')
    plt.xlabel('epoch')
    plt.ylabel('x')

# y 그래프 초기화
if show_y_graph:
    plt.figure("Y History")
    plt.title('Y History')
    plt.xlabel('epoch')
    plt.ylabel('y')

# step 그래프 초기화
if show_step_graph:
    figs, axes = plt.subplots(4, 1, figsize=(8, 16))

if save_info:
    sys.stdout = Logger('information_log.txt')

print(f'init x : {x}')
print(f'init y : {target_func(x)}')

# 최적화
for i, optim in enumerate(optims):

    # 동일한 위치에서 시작해야 하므로 새로운 옵티마이저마다 이전에 할당된 동일한 값을 옵티마이저가 계산할 x에 할당
    train_x = x
    history_x = [x]

    # 최적화 실행
    for epoch in range(epochs):
        # dx = num_diff2(target_func, train_x)  # 기울기 계산
        dx = grad_func(train_x)
        train_x = optim.update(train_x, dx)  # 기울기를 바탕으로 x 갱신

        history_x.append(train_x) # x history 업데이트

    # 최종적 x의 위치와 y의 위치 출력
    print(f'\nResult {optim.name}:')
    print(f"    last x : {train_x}")
    print(f"    last y : {target_func(train_x)}")

    # 옵티마이저 정보 출력
    if show_optim_info:
        print()
        for (info_name, info_value) in optim.__dict__.items():
            if info_name not in info_filter:
                print(f"    {info_name} : {info_value}")

    # x 그래프 업데이트
    if show_x_graph:
        plt.figure("X History")
        plt.plot(epoch_list, history_x, label=optim.name)

    # y 그래프 업데이트
    if show_y_graph:
        plt.figure("Y History")
        plt.plot(epoch_list, target_func(np.array(history_x)), label=optim.name)

    # step 그래프 업데이트
    if show_step_graph:
        x_list = np.arange(*bounds, graph_step)
        history_y = target_func(np.array(x_list))
        axes[i].scatter(history_x, target_func(np.array(history_x)), c=np.arange(len(history_x)), cmap='viridis', label='Path', s=10)
        axes[i].plot(x_list, history_y)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_title(optim.name, loc='right')

# x 그래프 라벨
if show_x_graph:
    plt.figure("X History")
    plt.legend()

    # x 그래프 저장
    if save_graph:
        plt.savefig(f"X_History.{save_type}")

# y 그래프 라벨
if show_y_graph:
    plt.figure("Y History")
    plt.legend()

    # y 그래프 저장
    if save_graph:
        plt.savefig(f"Y_History.{save_type}")

# step 그래프 저장
if show_step_graph and save_graph:
    figs.savefig(f'Step_History.{save_type}')

# 그래프 출력
if any([show_x_graph, show_y_graph, show_step_graph]):
    plt.show()
