1.프로그램 실행 단계：
==============
라즈베리파이에는 파이썬, PC는 파이토치를 각각 설치해 신경망 작동 환경을 제공한다.      

(1)将cloud文件夹放到PC机上，运行initCloud.py文件，启动服务端。   

![image](https://github.com/wyc941012/Edge-Intelligence/blob/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8D%8F%E5%90%8C%E6%8E%A8%E6%96%AD%E4%BB%BF%E7%9C%9F%E7%B3%BB%E7%BB%9F/image/cloud.jpg)  
(2)모바일 폴더를 라즈베리파이에 올려놓고 initMobile.py파일을 실행해 모델명과 최대 허용요금을 받는다.프로그램은 네트워크 속도, 비용 등의 요인에 따라 최적의 의사결정을 하고, 신경망 모델의 각 계층에서 최적의 실행 위치(모바일 또는 클라우드)를 얻으며, 프로그램은 오프로드 결정에 따라 컴퓨팅 작업을 할당하고, 신경망 모델의 추론 계산을 완성한다.
  
将mobile文件夹放到树莓派上，运行initMobile.py文件，传入参数分别为模型名称和最大可接受费用。程序将根据网速、费用等因素进行最优决策，得到神经网络模型每一层的最优运行位置(在移动端或者云端)，程序根据卸载决策分配计算任务，完成神经网络模型的推断计算。      

![image](https://github.com/wyc941012/Edge-Intelligence/blob/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8D%8F%E5%90%8C%E6%8E%A8%E6%96%AD%E4%BB%BF%E7%9C%9F%E7%B3%BB%E7%BB%9F/image/mobile.jpg)     

2.프로그램 설명：
=============
클라우드는 클라우드, 모바일은 모바일로 프로그램을 실행하는데 PC를 사용해 클라우드를, 라즈베리파이는 모바일을 시뮬레이션한다.
  
cloud和mobile目录分别代表云端和移动端运行程序，使用PC机仿真云端，树莓派仿真移动端。      

본 실험은 cifar-10 데이터 세트에서 AlexNet과 VGG16 모델을 훈련하였다.datasets 디렉토리는 cifar-10 데이터셋을 저장하고, model 디렉토리는 훈련된 CNN 모델을 저장한다.프로그램은 PyTorch 프레임워크의 특성을 살려 모델의 forward 방법을 개서하고 모델을 훈련시킨 후 레이어를 입도로 계산 태스크를 실행하여 모델의 계층적 추정을 실현한다.

本实验在cifar-10数据集上训练了AlexNet和VGG16模型。其中datasets目录存放cifar-10数据集，model目录存放训练好的CNN模型。程序借助PyTorch框架的特性，改写模型的forward方法，训练模型后，以层为粒度运行计算任务，实现模型的分层推断。  

~~~
def forward(self, x, startLayer, endLayer, isTrain):
	if isTrain:
	x = self.features(x)
	x = x.view(x.size(0), 2*2*128)
	x = self.classifier(x)
else:
	if startLayer==endLayer:
		if startLayer==10:
			x = x.view(x.size(0), 2*2*128)
			x = self.classifier[startLayer-10](x)
		elif startLayer<10:
			x = self.features[startLayer](x)
		else:
			x = self.classifier[startLayer-10](x)
	else:
		for i in range(startLayer, endLayer+1):
			if i<10:
				x = self.features[i](x)
			elif i==10:
				x = x.view(x.size(0), 2*2*128)
				x = self.classifier[i-10](x)
			else:
				x = self.classifier[i-10](x)
return x
~~~
