import numpy
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D   若样本为三维数据时使用

class Perceptron:
   #初始化　dim_out个神经元构成的单层神经网络，输出为dim_in 维
    def __init__(self,dim_in):
        #两分类问题
        dim_out=1
        #随机初始化权值矩阵　weight
        weight =numpy.ones([dim_in,dim_out])
        weight[1]=-10*weight[1]
        #初始化神经元的偏置bias
        bias=0
        #初始化增广的权值矩阵
        self.weightn=numpy.insert(weight,0,values=bias,axis=0)
        print(self.weightn)
        #分类完成标识符
        self.flag=0
        
    #学习
    def learn(self,xn,rate,margin):
        y=numpy.dot(xn,self.weightn)
        #寻找被错误分类的样本
        errors=y<margin
        if((errors==False).all()or(errors==True).all()):
            return 1
        xerr=-xn[numpy.where(y<0)[0],:]
        for j in range(len(xerr)):
            #梯度下降
            self.weightn=self.weightn+rate*xerr[j,:].reshape(-1,1)
            #print(self.weightn)
        return 0
            
if __name__=='__main__':
    #创建样本
    x = numpy.array(numpy.random.rand(100,2))
    x[:60,0]=-60*x[:60,0]
    x[60:,0]=60*x[60:,0]
    x[:60,1]=60*x[:60,1]
    x[60:,1]=60*x[60:,1]
    #初始化图片编号
    i=1
    for rate in [1e-4,1e-3,1e-2,1e-1,9e-1]:     #学习率
        #初始化图片编号
        j=1
        for add_separation in [0,1,2,3,4]:     #样本附加分离量
            #实例化感知器对象
            Robin = Perceptron(2)
            x[:60,0]=x[:60,0]-add_separation
            x[60:,0]=x[60:,0]+add_separation
            y = numpy.array([numpy.zeros(60),numpy.ones(60)]).reshape(120,1)
            #增广样本
            xn=numpy.insert(x,0,1,axis=1)  
            #规划化样本
            xn[:60,:]=-xn[:60,:]        
            #初始化迭代次数
            times=0
            #初始化稳定裕度
            margin=0.5
            for k in range(100000):
                if(Robin.learn(xn,rate,margin)==0):
                    times=times+1
                else:
                    break
            plt.plot(x[:60,0],x[:60,1],"bo")
            plt.plot(x[60:,0],x[60:,1],"ro")
            xs = numpy.linspace(-60,60,121)
            ys = (margin-Robin.weightn[0]-xs*Robin.weightn[1]) / Robin.weightn[2]
            print(Robin.weightn)
            print("迭代次数为：%d"%(times));
            plt.plot(xs, ys, "k--")
            #plt.show()   显示图片时使用
            plt.savefig('figure'+str(i)+'-'+str(j))
            plt.clf()
            #改变图片编号
            j=j+1
        #初始化附加样本分离量
        x[:60,0]=x[:60,0]+10
        x[60:,0]=x[60:,0]-10
        #改变图片编号
        i=i+1
        
 
