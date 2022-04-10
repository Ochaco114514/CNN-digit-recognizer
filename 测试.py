import csv
import numpy as np
import pandas as pd
import time


np.set_printoptions(threshold=np.inf)

def zero_pad(x,pad):
    x_pad= np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))#0填充长宽
    return x_pad

def rot180(x):
    return np.rot90(x,k=2)

def distribute_value(x,shape):
        (h,w)=shape
        average=x*1.0/(h*w)
        a=np.ones(shape)*average
        return a

def one_hot(y,num,k):
    ans=np.zeros((num,k))
    for i in range(num):
        ans[i][y[i]]=1
    return ans

def img2col(x,f,stride):
    h,w,c=x.shape
    H=(h-f)//stride+1
    img_col=np.zeros((H*H,f*f*c))   
    num=0
    for i in range(H):
        for j in range(H):
            img_col[num]=x[i*stride:i*stride+f, j*stride:j*stride+f, :].reshape(-1)
            num+=1
    return img_col

class Conv():
    """
    参数:
    kernel -- 卷积核shape
    stride -- 步长,默认1
    pad -- 0填充,默认不填充
    """
    def __init__(self,kernel,stride=1,pad=0) -> None:
        h,w,inc,outc=kernel
        self.kernel=kernel
        self.stride=stride
        self.pad=pad
        scale=np.sqrt(3*inc*w*h/outc)
        self.W=np.random.randn(h,w,inc,outc)/scale
        self.b=np.random.randn(outc)/scale
        self.W_gradient=np.zeros(kernel)
        self.b_gradient=np.zeros(outc)
        self.x=None
        self.img_col=None

    def reinit(self,W,b):
        self.W=W
        self.b=b

    def forward(self,x):
        pad=self.pad
        self.x=x
        if self.pad!=0:
            x=zero_pad(x,pad)
        m,h,w,c=x.shape
        f,f,inc,outc=self.W.shape
        H=(h-f)//self.stride+1
        feature=np.zeros((m,H,H,outc))  
        self.img_col=[]
        kernel=self.W.reshape(-1,outc)
        for i in range(m):
            image_col=img2col(x[i],f,self.stride)
            feature[i]=np.dot(image_col, kernel).reshape(H,H,outc)
            feature[i]=(np.dot(image_col, kernel)+self.b).reshape(H,H,outc)
            self.img_col.append(image_col)
        return feature

    def backward(self,delta,alpha):
        m,h,w,c=self.x.shape
        f,f,inc,outc=self.W.shape
        bd,wd,hd,cd=delta.shape

        delta_col=delta.reshape(bd,-1,cd)
        for i in range(m):
            self.W_gradient+=np.dot(self.img_col[i].T, delta_col[i]).reshape(self.W.shape)
        self.W_gradient/=m
        self.b_gradient+=np.sum(delta_col, axis=(0,1))
        self.b_gradient/=m

        delta_backward=np.zeros(self.x.shape)
        k_180=np.rot90(self.W,2,(0,1))
        k_180=k_180.swapaxes(2, 3)
        k_180_col=k_180.reshape(-1,c)

        if (hd-f+1)!=w:
            pad=(w-hd+f-1)//2
            pad_delta=zero_pad(delta,pad)
        else:
            pad_delta=delta

        for i in range(m):
            pad_delta_col=img2col(pad_delta[i],f,self.stride)
            delta_backward[i]=np.dot(pad_delta_col,k_180_col).reshape(h,w,c)

        self.W-=self.W_gradient*alpha
        self.b-=self.b_gradient*alpha

        return delta_backward
    
class Relu():
    def __init__(self) -> None:
        self.x=None

    def forward(self,x):
        self.x=x
        return np.maximum(x,0)
    
    def backward(self,delta):
        delta[self.x<0]=0
        return delta
    

class Pool():
    """
    参数:
    poolf -- 窗口大小,默认2,即2*2
    mode -- 池化类型,默认max,可选average
    """
    def __init__(self,poolf=2,mode="max") -> None:
        self.f=poolf
        self.mode=mode
        self.feature_mask=None

    def forward(self,x):
        m,h,w,c= x.shape
        feature_w=w//self.f
        feature=np.zeros((m,feature_w,feature_w,c))
        if self.mode=='max':
            self.feature_mask=np.zeros((m,h,w,c))
            for mi in range(m):
                for ci in range(c):
                    for i in range(feature_w):
                        for j in range(feature_w):
                            feature[mi,i,j,ci]=np.max(x[mi,i*self.f:i*self.f+self.f,j*self.f:j*self.f+self.f,ci])
                            index=np.argmax(x[mi,i*self.f:i*self.f+self.f,j*self.f:j*self.f+self.f,ci])
                            self.feature_mask[mi,i*self.f+index//self.f,j*self.f+index%self.f,ci]=1
        elif self.mode=="average":
            self.feature_mask=np.zeros((m,h,w,c))
            for mi in range(m):
                for ci in range(c):
                    for i in range(feature_w):
                        for j in range(feature_w):
                            feature[mi,i,j,ci]=np.mean(x[mi,i*self.f:i*self.f+self.f,j*self.f:j*self.f+self.f,ci])

        return feature

    def backward(self,delta):
        if self.mode=="max":
            return np.repeat(np.repeat(delta,self.f,axis=1),self.f,axis=2)*self.feature_mask
        elif self.mode=="average":
            m,h,w,c=delta.shape
            for mi in range(m):
                for ci in range(c):
                    for i in range(h):
                        for j in range(w):
                            self.feature_mask[mi,i*self.f:i*self.f+self.f,j*self.f:j*self.f+self.f,ci]+=distribute_value(delta[mi,i,j,ci],(self.f,self.f))
            return self.feature_mask

class NN():
    """
    参数:
    outc -- 输出维度,即隐藏层神经元数或输出层类数
    输入维度自动计算
    """
    def __init__(self,outc) -> None:
        self.outc=outc

    def initial(self,inc):
        outc=self.outc
        self.inc=inc
        scale=np.sqrt(inc/2)
        self.omega=np.random.randn(inc,outc)/scale
        self.bias=np.random.randn(outc)/scale
        self.omega_gradient=np.zeros((inc,outc))
        self.bias_gradient=np.zeros(outc)
        self.x=None

    def reinit(self,omega,bias):
        self.omega=omega
        self.bias=bias

    def forward(self,x):
        self.x=x
        x_forward=np.dot(self.x,self.omega)+self.bias
        return x_forward

    def backward(self,delta,alpha):
        batch_size=self.x.shape[0]
        self.omega_gradient=np.dot(self.x.T, delta)/batch_size
        self.bias_gradient=np.sum(delta, axis=0)/batch_size
        delta_backward=np.dot(delta,self.omega.T)
        self.omega-=self.omega_gradient*alpha
        self.bias-=self.bias_gradient*alpha
        return delta_backward

class Softmax():
    def __init__(self) -> None:
        self.softmax=None

    def cal_loss(self,predict,y):
        batchsize=predict.shape[0]
        self.predict(predict)
        loss=0
        delta=np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i]=self.softmax[i]-y[i]
            loss-=np.sum(np.log(self.softmax[i])*y[i])
        loss/=batchsize
        return loss,delta

    def predict(self,predict):
        batchsize=predict.shape[0]
        self.softmax=np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp=predict[i]-np.max(predict[i])
            predict_tmp=np.exp(predict_tmp)
            self.softmax[i]=predict_tmp/np.sum(predict_tmp)
        return self.softmax

def fit(layers,saveaddr,x_train,y_train,k,learning_rate=0.01,batchsize=10,iter=10):
    """
    参数:
    layers -- 自定义神经网络结构,列表形式,最后是softmax
    saveaddr -- 参数存储地址
    x_train -- numpy数组,样本数量*长*宽*通道
    y_train -- numpy数组,样本数量*1
    k -- 类数
    learning_rate -- 学习率,默认0.01
    batchsize -- batch大小,默认10
    iter -- 迭代次数,默认10
    """
    m,h,w,c=x_train.shape
    x_train=x_train.reshape(m,-1)
    x_train=x_train*1.0/np.max(x_train)
    x_train=x_train.reshape(m,h,w,c)
    y_train=one_hot(y_train,m,k)

    Conv_num=0
    NN_num=0

    for epoch in range(iter):
        for i in range(0,m,batchsize):
            x=x_train[i:i+batchsize]
            y=y_train[i:i+batchsize]
            for j in range(len(layers)):
                layer=layers[j]
                if isinstance(layer,Conv):
                    x=layer.forward(x)
                    if epoch==0 and i==0:
                        Conv_num+=1
                elif isinstance(layer,Relu):
                    x=layer.forward(x)
                elif isinstance(layer,Pool):
                    x=layer.forward(x)
                    cache=x.shape
                elif isinstance(layer,NN):
                    x=x.reshape(batchsize,-1)
                    if epoch==0 and i==0:
                        layer.initial(x.shape[1])
                        NN_num+=1
                    x=layer.forward(x)
                elif isinstance(layer,Softmax):
                    loss,delta=layer.cal_loss(x,y)
                layers[j]=layer
            flag=0
            for j in reversed(range(len(layers))):
                layer=layers[j]
                if isinstance(layer,NN):
                    delta=layer.backward(delta,learning_rate)
                elif isinstance(layer,Pool):
                    if flag==0:
                        delta=delta.reshape(cache)
                        flag=1
                    delta=layer.backward(delta)
                elif isinstance(layer,Relu):
                    delta=layer.backward(delta)
                elif isinstance(layer,Conv):
                    delta=layer.backward(delta,learning_rate)
                layers[j]=layer
            
            print("Epoch-{}-{:05d}".format(str(epoch),i), ":", "loss:{:.4f}".format(loss))
        
        learning_rate*=0.95**(epoch+1)
        if epoch==0:
            W={}
            omega={}
            b=[0]*Conv_num
            bias=[0]*NN_num
        Conv_num=0
        NN_num=0
        for layer in layers:
            if isinstance(layer,NN):
                omega[NN_num]=layer.omega
                bias[NN_num]=layer.bias
                NN_num+=1
            elif isinstance(layer,Conv):
                W[Conv_num]=layer.W
                b[Conv_num]=layer.b
                Conv_num+=1

        np.savez(saveaddr,W=W,b=b,omega=omega,bias=bias)


def pred(x_test,y_test,addr,layers):
    """
    参数：
    x_test -- numpy数组,样本数量*长*宽*通道
    y_test -- numpy数组,样本数量*1
    addr -- 参数保存地址
    layers -- 自定义神经网络结构,列表形式,最后是softmax
    """
    r=np.load(addr,allow_pickle=True)
    m,h,w,c=x_test.shape
    x_test=x_test.reshape(m,-1)
    x_test=x_test*1.0/np.max(x_test)
    x_test=x_test.reshape(m,h,w,c)
    
    Conv_num=0
    NN_num=0
    c=r["W"].item()
    n=r["omega"].item()
    for i in range(len(layers)):
        layer=layers[i]
        if isinstance(layer,NN):
            layer.reinit(n[NN_num],r["bias"][NN_num])
            NN_num+=1
        elif isinstance(layer,Conv):
            layer.reinit(c[Conv_num],r["b"][Conv_num])
            Conv_num+=1
        layers[i]=layer

    print(1)
    num=0
    for i in range(m):
        print(i)
        x=x_test[i]
        x=x[np.newaxis,:]
        y=y_test[i][0]
        for layer in layers:
            if isinstance(layer,Conv):
                x=layer.forward(x)
            elif isinstance(layer,Relu):
                x=layer.forward(x)
            elif isinstance(layer,Pool):
                x=layer.forward(x)
            elif isinstance(layer,NN):
                x=x.reshape(1,-1)
                x=layer.forward(x)
            elif isinstance(layer,Softmax):
                x=layer.predict(x)

        if np.argmax(x)==y:
            num+=1

    print("准确率: ",num/m*100,"%")



if __name__ == '__main__':

    t0=time.time()
    csv_data=pd.read_csv("\\train.csv")
    x=csv_data.drop(columns=['label'])
    x_train=x.to_numpy()
    m=x_train.shape[0]
    x_train=x_train.reshape(m,28,28,1)

    y=csv_data['label']
    y_train=y.to_numpy()
    
    layers=[]
    conv1=Conv(kernel=(5, 5, 1, 6))
    layers.append(conv1)
    relu1=Relu()
    layers.append(relu1)
    pool1=Pool()
    layers.append(pool1)
    conv2=Conv(kernel=(5, 5, 6, 16))
    layers.append(conv2)
    relu2=Relu()
    layers.append(relu2)
    pool2=Pool()
    layers.append(pool2)
    nn1=NN(10)
    layers.append(nn1)
    nn2=NN(10)
    layers.append(nn2)
    softmax = Softmax()
    layers.append(softmax)

    
    saveaddr="\data.npz"
    fit(layers=layers,saveaddr=saveaddr,x_train=x_train,y_train=y_train,k=10)
    

    x_test=pd.read_csv("\\test.csv")
    x_test=x_test.to_numpy()
    m=x_test.shape[0]
    x_test=x_test.reshape(m,28,28,1)

    y_test=pd.read_csv("\标准答案.csv")
    y_test=y_test.drop(columns=["ImageId"])
    y_test=y_test.to_numpy()

    pred(x_test=x_test,y_test=y_test,addr=saveaddr,layers=layers)
    t1=time.time()
    print("耗时：{}s".format(t1-t0))


#①bias不更新 3.47h 97.88%  ②更新 3.42h 98.05% ③三层1*8*16*16 5.84h 98.38% ④1*8*16 4.45h 98.08% ⑤1*4*8 2.39h 97.49% ⑥1*6*16 NN10*10 3.47h 98.13%