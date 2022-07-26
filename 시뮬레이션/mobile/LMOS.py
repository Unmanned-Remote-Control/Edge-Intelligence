import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ALEXNET_MODEL_PATH = "model/alexnetlayermodel.pkl"
model = torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
model_layer_cnt = len(model.features) + len(model.classifier)


# 단위 변환용 함수
def Unit(unit):
    units = ['B', 'K', 'M', 'G', 'T']
    val = units.index(unit)
    if val <= 0:
        return 1
    else:
        return 1024 ** (val)


# 사용 가능 용량
def getAvailable(path):
    diskInfo = os.statvfs(path)
    available = diskInfo.f_bsize * diskInfo.f_bavail
    # 사용 가능 용량을 키로 바이트(M)로 표시
    available = available / Unit('M')
    return available


# Edge/Server Convolution Latency
def Conv_Latency(computation_amount, CPU_Cores, Processor_Speed):
    T = computation_amount / (CPU_Cores * Processor_Speed)
    return T


def Transmission_Latency(x, bandwidth):
    output_ = output_size(x)
    T_tx = output_ / bandwidth
    return T_tx


def Calaulating_model_size(x):
    total = 0
    depth_pre = 3
    x += 1
    if x > len(model.features):
        x_ = len(model.features)
        x = x - x_
    else:
        x_ = x
        x = 0

    for layer in model.features[:x_]:
        weights = 0
        biases = 0
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            weights = layer.out_channels * (layer.kernel_size[0] * layer.kernel_size[1]) * depth_pre
            biases = layer.out_channels
            depth_pre = layer.out_channels
        total += weights + biases

    if (x != 0):
        # 첫번째 fc layer는 전 레이어가 pulling 이므로 다르게 계산
        pooling = model.features[-1]
        layer_1 = model.classifier[0]
        total += layer_1.out_features * ((pooling.kernel_size ** 2) * depth_pre) + layer_1.out_features

        for layer in model.classifier[1:x]:
            weights = 0
            biases = 0
            if isinstance(layer, nn.Linear):
                weights += layer.out_features * layer.in_features
                biases += layer.out_features
            total += weights + biases

    # 메가바이트 단위
    total = (total * 32) / 8000000
    return total


def output_size(x):
    # convolution, maxpooling
    size = 32
    x += 1
    if x > len(model.features):
        x = x - len(model.features) - 2
        return model.classifier[x].out_features
    else:
        x_ = x
    depth_pre = 3
    for layer in model.features[:x_]:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            size = (size - layer.kernel_size[0] + 2 * layer.padding[0]) / layer.stride[0] + 1
            depth_pre = layer.out_channels
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            size = (size - layer.kernel_size) / layer.stride + 1

    return size * size * depth_pre


def F1(x_1):
    global model
    computation_amount_local = Calaulating_model_size(x_1)
    #print("layer " + str(x_1) + "에서의 model_size : " + str(computation_amount_local))
    # 모델 사이즈 알아내야함

    if computation_amount_local > getAvailable('/'):
        return -1

    computation_amount_server = Calaulating_model_size(model_layer_cnt) - computation_amount_local
    #print("computation_amount_server : " + str(computation_amount_server))

    # RPi4 - 64-bit quad-core Cortex-A72 processor 논문에선 quad-core 1.5 GHz processor -> 그러면 4, 1.5 ??
    # Server - 내 PC (AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz) -> 그러면 8, 3.59 ??
    # The RPi4 modules and the cloud server are connected to a Wi-Fi network providing a bandwidth of 10 Mbps 로 논문에 나와있는데 정해놓고 제공하는 방식인가 -> 그러면 10?
    return Conv_Latency(computation_amount_local, CPU_Cores=4, Processor_Speed=1500) \
           + Transmission_Latency(x_1, bandwidth=10) \
           + Conv_Latency(computation_amount_server, CPU_Cores=8, Processor_Speed=3590)


def F2(x_1):
    computation_amount_local = Calaulating_model_size(x_1)
    return computation_amount_local


def LMOS_Algorithm():
    y_arr = []

    y1_min = F1(0)
    y2_min = -F2(0)
    print("layer cnt:",model_layer_cnt)
    # 각 레이어 당 y값 및 y_ideal 구하기 
    for x1 in range(model_layer_cnt):  # 레이어 개수
        y1 = F1(x1)
        y2 = -F2(x1)

        if y1 == -1:  # available memory 초과인 경우
            break

        y_arr.append((y1, y2, x1))

        if y1 < y1_min:
            y1_min = y1

        if y2 < y2_min:
            y2_min = y2

    plot_x = []
    plot_y = []
    for temp_y in y_arr:
        plot_x.append(temp_y[0])
        plot_y.append(temp_y[1])
        plt.text(temp_y[0], temp_y[1], temp_y[2])

    plt.xlabel('Latency')
    plt.ylabel('-Memory Util')
    plt.plot(plot_x, plot_y, 'bo')
    plt.show()

    y_ideal = (y1_min, y2_min)

    #print("ideal:", y_ideal)

    # y_nadir 구하기 
    y_nadir = []
    for y in y_arr:
        if y[0] == y_ideal[0]:
            y_nadir.append(y)

    #print("nadir:", y_nadir)

    # 입실론 지정하기
    e2 = min(y_nadir, key=lambda temp_y: temp_y[0])[1]
    #print("epsilon:", e2)

    F = []

    while e2 > y_ideal[1]:
        y_candidate = list(filter(lambda temp_y: temp_y[1] < e2 , y_arr))
        # if y_candidate is empty?
        y_candidate_min = min(y_candidate, key=lambda temp_y: temp_y[0])
        f_e_solution_arr = list(filter(lambda temp_y: temp_y[0] == y_candidate_min[0], y_candidate))
        F = F + f_e_solution_arr
        e2 = min(f_e_solution_arr, key=lambda temp_y: temp_y[1])[1]

    # F에서 dominant한 것 제외
    final_answers = []
    for f in F:
        if f[0] == min(y_arr, key =lambda temp_y: temp_y[0]) and f[1] == min(y_arr,key = lambda temp_y: temp_y[1]):
            continue
        else:
            final_answers.append(f)
    final_answer = min(F, key=lambda temp_y: temp_y[0])

    return final_answer[2]


#if __name__ == "__main__":
    #print(LMOS_Algorithm())
    # print(F1(0))
    # print(F1(1))
    # print(F1(2))
    # print(F1(3))
    # print(F1(4))
    # print(F1(5))
    # print(F1(6))
    # print(F1(7))
    # print(F1(8))
    # print(F1(9))
    # print(F1(10))
    # print(F1(11))
    # print(F1(12))
    # print(F1(13))
