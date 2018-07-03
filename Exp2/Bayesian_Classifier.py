import numpy
import matplotlib
import matplotlib.pyplot as plt

class Bayesian_Classifier:

    def __init__(self,domain,prior_probability1,prior_probability2,risk):
        self.domain=domain
        self.prior_probability1=prior_probability1
        self.prior_probability2=prior_probability2
        self.risk=risk
        
    def calConProbability(self,sample1,sample2):
        self.mean1=numpy.mean(sample1)
        self.std1=numpy.std(sample1)
        self.mean2=numpy.mean(sample2)
        self.std2=numpy.std(sample2)
        
    def calLikelyhood(self):
        #Calculate Likelyhood
        self.likelyhood1=1/(numpy.sqrt(2*numpy.pi)*self.std1)*numpy.exp(-1*(self.domain-self.mean1)**2/(2*self.std1**2))
        self.likelyhood2=1/(numpy.sqrt(2*numpy.pi)*self.std2)*numpy.exp(-1*(self.domain-self.mean2)**2/(2*self.std2**2))
        #Plot Likelyhood
        plt.figure(1)
        plt.title('Likelyhood')
        plt.plot(self.domain,self.likelyhood1,'b')
        plt.plot(self.domain,self.likelyhood2,'r')
        
    def calPostProbability(self):
        #Calculate Posterior Probability
        self.posterior_probability1=self.prior_probability1*self.likelyhood1
        self.posterior_probability2=self.prior_probability2*self.likelyhood2
        posterior_probability_sum=self.posterior_probability1+self.posterior_probability2
        self.posterior_probability1=self.posterior_probability1/posterior_probability_sum
        self.posterior_probability2=self.posterior_probability2/posterior_probability_sum
        #Plot Posterior Probability
        plt.figure(2)
        plt.title('Posterior Probability Without Considering Risk')
        plt.plot(self.domain,self.posterior_probability1,'b')
        plt.plot(self.domain,self.posterior_probability2,'r')
        #Calculate Decision Boundary Without Considering Risk
        count=0
        while self.posterior_probability2[count]<0.5:
            count=count+1
        print("Decision Boundary Without Considering Risk:%",self.domain[count])
        
    def calPostProbabilityWithDecisionRisk(self):
        #Calculate Cost
        cost=numpy.dot(self.risk,numpy.array([[self.posterior_probability1],[self.posterior_probability2]]).reshape(2,101))
        #Plot Cost
        plt.figure(3)
        plt.title('Posterior Probability Considering Risk')
        plt.plot(self.domain,cost[0],'r')
        plt.plot(self.domain,cost[1],'b')
        #Calculate Decision Boundary Considering Risk
        count=0
        while cost[0,count]<cost[1,count]:
            count=count+1
        print("Decision Boundary Considering Risk:%",self.domain[count])
        
        
if __name__=='__main__':
    #Build Samples
    sample1=numpy.array([-3.9847, -3.5549, -1.2401, -0.9780, -0.7932, -2.8531, -2.7605, -3.7287,-3.5414,-2.2692,
    -3.4549,-3.0752,-3.9934, -0.9780,-1.5799,-1.4885, -0.7431,-0.4221,-1.1186,-2.3462,-1.0826,-3.4196,-1.3193,
    -0.8367,-0.6579,-2.9683])
    sample2=numpy.array([2.8792, 0.7932, 1.1882, 3.0682, 4.2532, 0.3271,0.9846,2.7648,2.6588])
    domain=numpy.arange(-5.0,5.1,0.1)
    risk=numpy.array([[0,1],[6,0]])
    prior_probability1=0.9
    prior_probability2=0.1
    #Classify
    bayesian_Classifier=Bayesian_Classifier(domain,prior_probability1,prior_probability2,risk)
    bayesian_Classifier.calConProbability(sample1,sample2)    
    bayesian_Classifier.calLikelyhood()
    bayesian_Classifier.calPostProbability()
    bayesian_Classifier.calPostProbabilityWithDecisionRisk()
    plt.show()
