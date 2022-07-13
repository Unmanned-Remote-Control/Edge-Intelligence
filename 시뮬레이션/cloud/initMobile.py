import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
from data import get_data_set
import socket
import threading
import pickle
import io
import sys
import time

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="model/alexnetlayermodel.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.123.10"
PORT=8081

class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def run(model, inputData, startLayer, endLayer):
	print("Mobile running %d to %d layer" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False) # model 함수는 어디서? 
	return outputs

def test(outputs, test_x, test_y):
	prediction = torch.max(outputs.data, 1) # 각 row에서 최댓값들 모은 텐서 반환 https://pytorch.org/docs/stable/generated/torch.max.html :이거 마지막 예시 참고
	correct_classified += np.sum(prediction[1].numpy() == test_y.numpy()) # .nump() : 넘파이 배열로 변환 , 예측한 것과 실제 값이 같은 것의 개수 반환하는 듯 
	acc=(correct_classified/len(test_x))*100 # 정확도 - 퍼센트로 반환 
	return acc

def sendData(client, inputData, startLayer, endLayer):
	data=Data(inputData, startLayer, endLayer) # Data 객체 생성 
	str=pickle.dumps(data) # 데이터 객체 자체를 파일로 저장 
	client.send(len(data).to_bytes(length=6, byteorder='big')) # client : 소켓 말하는듯 
	client.send(data)

def receiveData(client, model, x, test_x, test_y):
	while True:
		lengthData=client.recv(6) # client 소켓 - recieve함수 
		length=int.from_bytes(lengthData, byteorder='big') # 길이에 대한 데이터 받아옴 
		b=bytes()
		count=0
		while True: # 데이터 보낸 길이만큼 다 받아올때까지 loop(왜 이렇게 하지?)
			value=client.recv(length)
			b=b+value
			count+=len(value)
			if count>=length:
				break
		data=pickle.loads(b) # 객체 그대로 저장한 거 다시 불러오기 
		if data.startLayer>=len(x): # x의 의미는? 
			acc=test(outputs, test_x, test_y) # 데이터 넣어서 정확도 구하기 
			end=time.time()
			print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc)) # runtime이 있나? 
			client.close()
			break
		else:
			count=0
			for i in range(startLayer, len(x)): # data.startLayer 아니고? 
				if x[i]==1: # x배열에서 1인 게 나오면 break
					break
				count=i
			outputs=run(model, test_x, startLayer, count) # model 함수의 인자 =>  input data : test_x , 시작레이어 : startLayer, 끝 레이어 : count 
			if count==len(x)-1: # x의 길이가 전체 레이어의 개수인가? - 그러면 전체 레이어 를 모바일에서만 돌리는걸 말하나 ? 
				acc=test(outputs, test_x, test_y)
				end=time.time()
				print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc))
				client.close()
				break
			else:
				endLayer=0
				for i in range(count+1, len(x)): # x[i]가 0인 부분 나오면 중단하고 뒷 레이어 전부 보내버리는 듯 
					if x[i]==0:
						break
					endLayer=i
				sendData(client, outputs, count+1, endLayer)

if __name__=="__main__":
	model=torch.load(ALEXNET_MODEL_PATH, map_location='cpu') # 모델 불러오기 
	device = torch.device("cpu")
	torch.set_num_threads(3)
	test_x,test_y,test_l=get_data_set("test")  # data의 함수 : test 데이터에 해당하는 데이터 x, y 값 가져오기 
	test_x=torch.from_numpy(test_x[0:100]).float()
	test_y=torch.from_numpy(test_y[0:100]).long()
	print("Model loaded successfully.")
	client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client.connect((IP, PORT))
	print("Cloud connection successful, ready for computing mission")
	print("Task submitted. Unload decision made")
	x=[1,1,1,1,1,1,1,1,1,1,1,1,1] # 레이어 개수가 아닌가본데 - 13 개 (alex net 은 8개 레이어, vgg 는  conv 13개, fc 3개 )
	print("Start running computational tasks")
	start=time.time()
	if x[0]==1: # x의 0번째 값이 1 인 경우 
		count=0
		for i in range(1, len(x)):
			if x[i]==0:
				break
			count=i
		sendData(client, outputs, 0, count) # x == 0 인 부분에서 중단하고 데이터 보내기 
		t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
		t.start()

	else: # x의 0번째 값이 1이 아닌 경우 
		count=0
		for i in range(1, len(x)):
			if x[i]==1:
				break
			count=i
		outputs=run(model, test_x, 0, count) # x가 1인 부분에서 멈추고 run 
		if count==len(x)-1:
			acc=test(outputs, test_x, test_y)
			end=time.time()
			print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc))
			client.close()
		else:
			endLayer=0
			for i in range(count+1, len(x)):
				if x[i]==0:
					break
				endLayer=i
			sendData(client, outputs, count+1, endLayer)
			t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
			t.start()




