import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib import cm

class Bayesian_Classifier_MultiDim:
    def __init__(self,domainx,domainy,prior_probability1,prior_probability2,prior_probability3,risk):
        self.domainX,self.domainY=numpy.meshgrid(domainx,domainy)
        self.prior_probability1=prior_probability1
        self.prior_probability2=prior_probability2
        self.prior_probability3=prior_probability3
        self.risk=risk
        
    def calConProbability(self,sample1,sample2,sample3):
        self.mean1=numpy.mean(sample1,axis=1)
        self.cov1=numpy.cov(sample1)
        self.mean2=numpy.mean(sample2,axis=1)
        self.cov2=numpy.cov(sample2)
        self.mean3=numpy.mean(sample3,axis=1)
        self.cov3=numpy.cov(sample3)
        
    def calLikelyhood(self):
        #Calculate Likelyhood
        pos =numpy.empty(self.domainX.shape+(2,))
        pos[:,:,0]=self.domainX;
        pos[:,:,1]=self.domainY;
        self.likelyhood1=multivariate_normal.pdf(pos,mean=self.mean1,cov=self.cov1)
        self.likelyhood2=multivariate_normal.pdf(pos,mean=self.mean2,cov=self.cov2)
        self.likelyhood3=multivariate_normal.pdf(pos,mean=self.mean3,cov=self.cov3)
        
        fig1=plt.figure(1)
        ax1=Axes3D(fig1)
        plt.title('Likelyhood')
        ax1.plot_surface(self.domainX, self.domainY, self.likelyhood1, color='r',linewidth=0, antialiased=False)
        ax1.plot_surface(self.domainX, self.domainY, self.likelyhood2,color=(151/255,251/255,152/255),linewidth=0, antialiased=False)
        ax1.plot_surface(self.domainX, self.domainY, self.likelyhood3, color=(153/255,50/255,204/255),linewidth=0, antialiased=False)
        
    def calPostProbability(self):
        #Calculate Posterior Probability
        self.posterior_probability1=self.prior_probability1*self.likelyhood1
        self.posterior_probability2=self.prior_probability2*self.likelyhood2
        self.posterior_probability3=self.prior_probability3*self.likelyhood3
        posterior_probability_sum=self.posterior_probability1+self.posterior_probability2+self.posterior_probability3
        self.posterior_probability1=self.posterior_probability1/posterior_probability_sum
        self.posterior_probability2=self.posterior_probability2/posterior_probability_sum
        self.posterior_probability3=self.posterior_probability3/posterior_probability_sum
        
        fig2=plt.figure(2)
        ax2=Axes3D(fig2)
        plt.title('Posterior Probability Without Considering Risk')
        ax2.plot_surface(self.domainX, self.domainY, self.posterior_probability1, color='r',linewidth=0, antialiased=False)
        ax2.plot_surface(self.domainX, self.domainY, self.posterior_probability2, color=(151/255,251/255,152/255),linewidth=0, antialiased=False)
        ax2.plot_surface(self.domainX, self.domainY, self.posterior_probability3, color=(153/255,50/255,204/255),linewidth=0, antialiased=False)

    def calDecisionBoundary(self):
        ##Calculate Decision Boundary
        region=numpy.zeros([numpy.size(self.domainX,axis=0),numpy.size(self.domainX,axis=1)])
        for i in range(numpy.size(self.domainX,axis=0)):
            for j in range(numpy.size(self.domainX,axis=1)):
                region[i,j]=numpy.argmax([self.posterior_probability1[i,j],self.posterior_probability2[i,j],self.posterior_probability3[i,j]])
        fig3=plt.figure(3)
        ax3=Axes3D(fig3)
        plt.title('Decision Boundary')
        ax3.view_init(elev=90,azim=0)
        surf3=ax3.plot_surface(self.domainX, self.domainY, region, cmap='rainbow',linewidth=0, antialiased=False)
        fig3.colorbar(surf3, shrink=0.5, aspect=5)
    
                
if __name__=='__main__':
    #build samples
    mean1=numpy.array([2,2])
    mean2=numpy.array([0,0])
    mean3=numpy.array([-2,-2])
    cov1=numpy.array([[1,0],[0,9]])
    cov2=numpy.array([[4,0],[0,16]])
    cov3=numpy.array([[4,0],[0,9]])
    sample1=numpy.random.multivariate_normal(mean1, cov1, 100).T
    sample2=numpy.random.multivariate_normal(mean2, cov2, 100).T
    sample3=numpy.random.multivariate_normal(mean3, cov3, 100).T
    
    domainx=numpy.arange(-10.0,10.0,0.05)
    domainy=numpy.arange(-10.0,10.0,0.05)
    
    prior_probability1=0.7
    prior_probability2=0.25
    prior_probability3=0.05
    
    risk=numpy.array([[0,3,7],[8,0,8],[0,2,6]])
    
    bayesian_Classifier_MultiDim=Bayesian_Classifier_MultiDim(domainx,domainy,prior_probability1,prior_probability2,prior_probability3,0.1)
    bayesian_Classifier_MultiDim.calConProbability(sample1,sample2,sample3)
    bayesian_Classifier_MultiDim.calLikelyhood()
    bayesian_Classifier_MultiDim.calPostProbability()
    bayesian_Classifier_MultiDim.calDecisionBoundary()
    plt.show()
