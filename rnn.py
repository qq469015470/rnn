import torch
import torch.nn as nn
import random
import torch.optim as optim
import math
import numpy as np

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#参数以次输入x为2维,一个隐含层16个单元(即1x16矩阵),1层隐含层,输入的第一个参数为Batch(Batch,Seqlen,x)否则为(Seqlen,Batch,x)
		self.rnn = nn.RNN(input_size=2, hidden_size=16, num_layers=1,  batch_first=True)
		self.liner = torch.nn.Linear(16, 1)#输出层,1维,因为最后的结果为一个二进制数

	def forward(self, x, hidden_prev):
		out, hidden_prev = self.rnn(x, hidden_prev)
		out = torch.relu(out)
		out = self.liner(out)
		out = torch.sigmoid(out)#使用sigmoid让y处于0-1之间
		return out, hidden_prev

model = Net()
optimizer = optim.Adam(model.parameters(), 0.01)
criterion= nn.MSELoss()#一般来说回归问题用均方差损失函数
#criterion= nn.CrossEntropyLoss()

#int转二进制tensor张量
def int2binary(num):
	temp = torch.zeros(10)
	for i in range(temp.size(0)):
		temp[i] = int(num % 2)
		num = num / 2

	return temp

#张量转换int
def binary2int(binary):
	num = 0
	for i in range(binary.size(0)):
		digit = round(binary[i].item())
		num += digit * pow(2, i)
	
	return num

def train():
	EPOCH = 100000
	hidden_prev = torch.zeros(1, 1, 16)
	for i in range(EPOCH):
		a = int(random.random() * 512)#这里做10位2进制的加法相加后可能会变11位,所以减一位随机0-512(2的9次方)
		b = int(random.random() * 512)
		target_bin = int2binary(a+b)#相加结果的二进制
		
		x = torch.zeros(1, 10, 2)#1个batch,10位二进制,取出a b 每个二进制数,所以为2维
		for j in range(target_bin.size(0)):
			x[0][j][0] = int2binary(a)[j]
			x[0][j][1] = int2binary(b)[j]

		hidden_prev = hidden_prev.detach()#隐含层不做反向传播,不这样会报错,原因不确定大概因为每次这个变量都被覆盖了反向传播时,找不到上次的变量因为已经被释放
		output, hidden_prev = model(x, hidden_prev)#rnn的输出与输入的时间序一致这里是(batch=1,seqlen=10,最后连接了输出层=1)

		#升维降维用于计算损失函数
		output = output.squeeze(2)#降维从(1,10,1)变为(1,10)
		target_bin = target_bin.unsqueeze(0)#升维从(10)变(1,10)

		loss = criterion(output, target_bin)
		model.zero_grad()#清空累计梯度
		loss.backward()#计算梯度
		optimizer.step()#更新权值


		if i % 1000 == 0:
			print(output)
			print(target_bin)
			print("time%d: loss:%f" % (i / 1000, loss), end='  ')
			print("%d + %d = %d" % (a, b, binary2int(output[0])), end='  ')
			print("%d + %d = %d" % (a, b, a+b))


train()
