import time
import os
import struct
import gzip
from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.conv import Conv2D
from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerMultiProcess
from pyvqnet.nn import activation as F
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor
import pyqpanda as pq

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pyqpanda import *
import pyqpanda as pq
import numpy as np
try:
    matplotlib.use("TkAgg")
except:  #pylint:disable=bare-except
    print("Can not use matplot TkAgg")
    pass

try:
    import urllib.request
except ImportError:
    raise ImportError("You should use Python 3.x")

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def _download(dataset_dir,file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as f:
            file_path_ungz = file_path[:-3].replace('\\', '/')
            if not os.path.exists(file_path_ungz):
                open(file_path_ungz,"wb").write(f.read())
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
    print("Done")

def download_mnist(dataset_dir):
    for v in key_file.values():
        _download(dataset_dir,v)

def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
    import os, struct
    from array import array as pyarray
    download_mnist(path)
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def data_select(train_num, test_num):
    x_train, y_train = load_mnist("training_data")
    x_test, y_test = load_mnist("testing_data")
    # Train Leaving only labels 0 and 1
    idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                    np.where(y_train == 1)[0][:train_num])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    x_train = x_train / 255
    y_train = np.eye(2)[y_train].reshape(-1, 2)
    # Test Leaving only labels 0 and 1
    idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                    np.where(y_test == 1)[0][:test_num])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    x_test = x_test / 255
    y_test = np.eye(2)[y_test].reshape(-1, 2)
    return x_train, y_train, x_test, y_test

n_samples_show = 6

x_train, y_train, x_test, y_test = data_select(100, 50)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

for img ,targets in zip(x_test,y_test):
    if n_samples_show <= 3:
        break

    if targets[0] == 1:
        axes[n_samples_show - 1].set_title("Labeled: 0")
        axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        n_samples_show -= 1

for img ,targets in zip(x_test,y_test):
    if n_samples_show <= 0:
        break

    if targets[0] == 0:
        axes[n_samples_show - 1].set_title("Labeled: 1")
        axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        n_samples_show -= 1
plt.show()
def qcnn_circuit(x ,weights, num_qubits, num_clist):
    machine = pq.MPSQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)
    cir = pq.QProg()

    for i in range(num_qubits):
        cir.insert(pq.H(qubits[i]))
        cir.insert(pq.RZ(qubits[i],x[i]))

    for i in range(num_qubits):
        cir.insert(pq.RY(qubits[i],weights[i]))
        cir.insert(pq.RZ(qubits[i],weights[4+i]))

    for i in range(num_qubits-1):
        cir.insert(pq.CNOT(qubits[i],qubits[i+1]))
    cir.insert(pq.CNOT(qubits[num_qubits-1],qubits[0]))

    for i in range(num_qubits):
        cir.insert(pq.RY(qubits[i],weights[8+i]))
        cir.insert(pq.RZ(qubits[i],weights[12+i]))

    result0 = machine.prob_run_list(cir, [qubits[0]], -1)
    result1 = machine.prob_run_list(cir, [qubits[1]], -1)
    result2 = machine.prob_run_list(cir, [qubits[2]], -1)
    result3 = machine.prob_run_list(cir, [qubits[3]], -1)

    result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
    machine.finalize()
    return result

def build_multiprocess_qmlp_circuit(x, weights, num_qubits, num_clist):
    out = np.zeros((196))
    t=0
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = qcnn_circuit(
                [
                    x[j*28+k],x[j*28+k+1],
                    x[(j+1)*28+k],x[(j+1)*28+k+1]
                ],
                weights,num_qubits,num_clist
            )
            out[t] = q_results[0]
            t=t+1
    return out
def qtcn_circuit(x ,weights, num_qubits, num_clist):
    machine = pq.MPSQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)
    cir = pq.QProg()

    for i in range(num_qubits):
        cir.insert(pq.H(qubits[i]))
        cir.insert(pq.RZ(qubits[i],x[i]))

    for i in range(num_qubits):
        cir.insert(pq.RY(qubits[i],weights[i]))
        cir.insert(pq.RZ(qubits[i],weights[4+i]))

    for i in range(num_qubits-1):
        cir.insert(pq.CNOT(qubits[i],qubits[i+1]))
    cir.insert(pq.CNOT(qubits[num_qubits-1],qubits[0]))

    for i in range(num_qubits):
        cir.insert(pq.RY(qubits[i],weights[8+i]))
        cir.insert(pq.RZ(qubits[i],weights[12+i]))

    result0 = machine.prob_run_list(cir, [qubits[0]], -1)
    result1 = machine.prob_run_list(cir, [qubits[1]], -1)
    result2 = machine.prob_run_list(cir, [qubits[2]], -1)
    result3 = machine.prob_run_list(cir, [qubits[3]], -1)

    result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
    machine.finalize()
    return result

def build_multiprocess_qmlp_circuit(x, weights, num_qubits, num_clist):
    out = np.zeros((196))
    t=0
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = qtcn_circuit(
                [
                    x[j*28+k],x[j*28+k+1],
                    x[(j+1)*28+k],x[(j+1)*28+k+1]
                ],
                weights,num_qubits,num_clist
            )
            out[t] = q_results[0]
            t=t+1
    return out
#模型定义
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quantum_circuit = QuantumLayerMultiProcess(build_multiprocess_qmlp_circuit,4*4,4, 1, diff_method="finite_diff")
        self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
        self.fc1 = Linear(input_channels=196, output_channels=10)
        self.fc2 = Linear(input_channels=10, output_channels=2)

    def forward(self, x):
        x = tensor.flatten(x, 1)
        x = self.quantum_circuit(x)     # 1 1 14 14
#         x = self.maxpool1(x)         # 1 1 7 7
        x = tensor.flatten(x, 1)   # 1 196
        x = F.ReLu()(self.fc1(x))  # 1 10
        x = self.fc2(x)            # 1 2
        return x
x_train, y_train, x_test, y_test = data_select(100, 50)
#实例化
model = Net()
#使用Adam完成此任务就足够了，model.parameters（）是模型需要计算的参数。
optimizer = Adam(model.parameters(), lr=0.005)
#分类任务使用交叉熵函数
loss_func = CategoricalCrossEntropy()

#训练次数
epochs = 20
train_loss_list = []
val_loss_list = []
train_acc_list =[]
val_acc_list = []

u=1
for epoch in range(1, epochs):
    total_loss = []
    model.train()
    batch_size = 1
    correct = 0
    n_train = 0
    w = 0
    for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(y, output)
        loss_np = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_train += batch_size
        loss.backward()
        optimizer._step()
        total_loss.append(loss_np)
        w=w+1
        print("第",u,"轮，第",w,"张数据")
    u=u+1

    train_loss_list.append(np.sum(total_loss) / len(total_loss))
    train_acc_list.append(np.sum(correct) / n_train)
    print("{:.0f} loss is : {:.10f}".format(epoch, train_loss_list[-1]))

    model.eval()
    correct = 0
    n_eval = 0

    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        loss = loss_func(y, output)
        loss_np = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_eval += 1

        total_loss.append(loss_np)
    print(f"Eval Accuracy: {correct / n_eval}")
    val_loss_list.append(np.sum(total_loss) / len(total_loss))
    val_acc_list.append(np.sum(correct) / n_eval)
import os
plt.figure()
xrange = range(1,len(train_loss_list)+1)
figure_path = os.path.join(os.getcwd(), 'HQCNN改_01 LOSS.png')
plt.plot(xrange,train_loss_list, color="blue", label="train")
plt.plot(xrange,val_loss_list, color="red", label="validation")
plt.title('HQCNN')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(1, epochs +1,step = 2))
plt.legend(loc="upper right")
plt.savefig(figure_path)
plt.show()

plt.figure()
figure_path = os.path.join(os.getcwd(), 'HQCNN改_01 Accuracy.png')
plt.plot(xrange,train_acc_list, color="blue", label="train")
plt.plot(xrange,val_acc_list, color="red", label="validation")
plt.title('HQCNN')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, epochs +1,step = 2))
plt.legend(loc="lower right")
plt.savefig(figure_path)
plt.show()
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
model.eval()
for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
    if count == n_samples_show:
        break
    x = x.reshape(-1, 1, 28, 28)
    output = model(x)
    pred = QTensor.argmax(output, [1],False)
    axes[count].imshow(x[0].squeeze(), cmap='gray')
    axes[count].set_xticks([])
    axes[count].set_yticks([])
    axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
    count += 1
plt.show()