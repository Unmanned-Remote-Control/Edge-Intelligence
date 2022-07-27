import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ALEXNET_MODEL_PATH = "model/alexnetlayermodel.pkl"
model = torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
model_layer_cnt = len(model.features) + len(model.classifier)

bandwidth = 100
plt_color= "r"
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
    T_tx = output_ / (bandwidth*1000000)
    return T_tx


def Calaulating_model_size(x):
    total = 0
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
            weights = layer.out_channels * (layer.kernel_size[0] * layer.kernel_size[1]) * layer.in_channels
            biases = layer.out_channels

            total += weights + biases

    for layer in model.classifier[:x]:
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
    # convolution, pooling
    size = 224 #나중에 실제 input 이미지 고려 바꿔야함
    x += 1
    if x > len(model.features):
        x = x - len(model.features) - 2
        return model.classifier[x].out_features * 32    #bit로만 반환
    else:
        x_ = x

    depth_pre = 3
    for layer in model.features[:x_]:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            size = (size - layer.kernel_size[0] + 2 * layer.padding[0]) / layer.stride[0] + 1
            depth_pre = layer.out_channels
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            size = (size - layer.kernel_size) / layer.stride + 1

    return (size * size * depth_pre * 32)   #bit로만 반환


def F1(x_1):
    global model
    computation_amount_local = Calaulating_model_size(x_1)

    #라즈베리파이 사용 가능 용량
    '''if computation_amount_local > getAvailable('/'):
        return -1'''

    computation_amount_server = Calaulating_model_size(model_layer_cnt) - computation_amount_local

    #테스트 출력
    '''print("layer : ", x_1)
    print("edge conv : ",Conv_Latency(computation_amount_local, CPU_Cores=4, Processor_Speed=1500))
    print("Transmission_Latency : ",Transmission_Latency(x_1, bandwidth))
    print("server conv : ",Conv_Latency(computation_amount_server, CPU_Cores=8, Processor_Speed=3590))
    print("total : ", Conv_Latency(computation_amount_local, CPU_Cores=4, Processor_Speed=1500) \
           + Transmission_Latency(x_1, bandwidth) \
           + Conv_Latency(computation_amount_server, CPU_Cores=8, Processor_Speed=3590))'''

    # RPi4 - 64-bit quad-core Cortex-A72 processor 논문에선 quad-core 1.5 GHz processor -> 그러면 4, 1.5 ??
    # Server - 내 PC (AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz) -> 그러면 8, 3.59 ??
    # The RPi4 modules and the cloud server are connected to a Wi-Fi network providing a bandwidth of 10 Mbps 로 논문에 나와있는데 정해놓고 제공하는 방식인가 -> 그러면 10?
    return Conv_Latency(computation_amount_local, CPU_Cores=4, Processor_Speed=1500) \
           + Transmission_Latency(x_1, bandwidth) \
           + Conv_Latency(computation_amount_server, CPU_Cores=8, Processor_Speed=3590)


def F2(x_1):
    computation_amount_local = Calaulating_model_size(x_1)
    return computation_amount_local


def LMOS_Algorithm():
    y_arr = []

    y1_min = F1(0)
    y2_min = -1 * F2(0)

    # 각 레이어 당 y값 및 y_ideal 구하기
    for x1 in range(model_layer_cnt):  # 레이어 개수
        y1 = F1(x1)
        y2 = -1 * F2(x1)

        if y1 == -1:  # available memory 초과인 경우
            break

        y_arr.append((y1, y2, x1))

        if y1 < y1_min:
            y1_min = y1

        if y2 < y2_min:
            y2_min = y2
    if len(y_arr)==0 :
        return -1
    elif len(y_arr)==1:
        return 0
    plot_x = []
    plot_y = []
    for temp_y in y_arr:
        plot_x.append(temp_y[0])
        plot_y.append(-1 * temp_y[1])
        plt.text(temp_y[0], -1 * temp_y[1], temp_y[2])


    plt.xlabel('Latency')
    plt.ylabel('Memory Util')
    plt.plot(plot_x, plot_y,color = plt_color ,marker='o', linestyle="-")
    plt.savefig('bandwidth_'+str(bandwidth)+'.png')
    #plt.clf()
    #plt.show()

    #도미넌트한 결과 제거
    y_ideal = (y1_min, y2_min)
    for f in y_arr:
        if f[0] == min(y_arr, key =lambda temp_y: temp_y[0]) and f[1] == min(y_arr,key = lambda temp_y: temp_y[1]):
            y_arr.remove(f)

    #그래프 그리는 부분
    plot_x = []
    plot_y = []
    for temp_y in y_arr:
        plot_x.append(temp_y[0])
        plot_y.append(-1 * temp_y[1])
        plt.text(temp_y[0], -1 * temp_y[1], temp_y[2])

    plt.xlabel('Latency')
    plt.ylabel('Memory Util')
    plt.plot(plot_x, plot_y, color=plt_color, marker='o', linestyle="-")
    plt.savefig('bandwidth_' + str(bandwidth) + '.png')
    # plt.clf()
    # plt.show()

    # y_nadir 구하기 - 원래 코드
    '''
    y_nadir = []
    for y in y_arr:
        if y[0] == y_ideal[0]:
            y_nadir.append(y)

    # 입실론 지정하기
    e2 = min(y_nadir, key=lambda temp_y: temp_y[0])[1]
    '''

    # y_nadir 구하기
    y1_nadir = 100
    y2_nadir = 0
    for y in y_arr:
        if y[0] == y_ideal[0]:
            if y2_nadir > y[1]:
                y2_nadir = y[1]
        elif y[1] == y_ideal[1]:
            if y1_nadir > y[0]:
                y1_nadir = y[0]
    y_nadir = (y1_nadir, y2_nadir)

    # 입실론 지정하기
    e2 = y_nadir[1]

    F = []
    while e2 > y_ideal[1]:
        y_candidate = list(filter(lambda temp_y: temp_y[1] <= e2, y_arr))
        y_candidate_min = min(y_candidate, key=lambda temp_y: temp_y[0])
        f_e_solution_arr = list(filter(lambda temp_y: temp_y[0] == y_candidate_min[0], y_candidate))
        F = F + f_e_solution_arr
        e2 = min(f_e_solution_arr, key=lambda temp_y: temp_y[1])[1]

    # F에서 dominant한 것 제외
    final_answers = []
    for f in F:
        if f[0] == min(y_arr, key =lambda temp_y: temp_y[0]) and f[1] == min(y_arr,key = lambda temp_y: temp_y[1]):
            print("dominant", f)
            continue
        else:
            final_answers.append(f)

    final_answer = min(F, key=lambda temp_y: temp_y[0])

    return final_answer[2]


if __name__ == "__main__":
    '''
    colors = ["lightcoral", "darkorange","green", "lime", "navy", "purple", "olive","indigo","steelblue","grey"]
    idx = 0
    for i in range(model_layer_cnt):
        computation_amount_local = Calaulating_model_size(i)
        computation_amount_server = Calaulating_model_size(model_layer_cnt) - computation_amount_local
        print("layer : ",i)
        print("Conv_Latency_Edge : "+ str(Conv_Latency(computation_amount_local, CPU_Cores=4, Processor_Speed=1500)))
        print("Transmission_Latency : "+ str(Transmission_Latency(i, bandwidth)))
        print("Conv_Latency_Server : "+ str(Conv_Latency(computation_amount_server, CPU_Cores=8, Processor_Speed=3590)))
        print("memory_usage : -"+ str(Calaulating_model_size(i)))
        print('----------------------------------------------------------------------------')
    '''
    #bandwidth = 10
    #print(LMOS_Algorithm())
    for i in range(1,20) :
        bandwidth = i * 5
        print(LMOS_Algorithm())


    '''
    for i in range(11) :
        if i == 0:
            i = 0.5
        bandwidth = i * 20
        plt_color = colors[idx]
        idx+=1
        print(LMOS_Algorithm())
    '''
