class LogisticRegression:
    def __init__(self,data,bias=1,alpha=0.01,target="target"):
        import numpy as np
        self.data = data
        self.target_name = target
        self.target = np.array(data[target])
        self.data = data.drop(columns=target)
        self.data = self.data.reset_index(drop=True)
        self.bias = bias 
        self.alpha = alpha 
        self.theta_vector = np.zeros(self.data.shape[1] + 1)
        self.m = len(self.data)
    def Tx(self,sample_n=None):
        import numpy as np 
        x_features = self.get_xfeatures_ytarget(sample_n=sample_n)[0]
        Tx = np.dot(self.theta_vector,x_features)
        return Tx 
    def activation(self,Tx):
        import numpy as np
        return 1 / (1 + np.exp(-1*Tx))
    def get_xfeatures_ytarget(self, sample_n):
        import numpy as np 
        x_features = np.insert(self.data.iloc[sample_n,:],0,self.bias)
        y_target = self.target[sample_n]
        return (x_features,y_target)
    def J(self,sample_n):
        import numpy as np
        x_features,y_i = self.get_xfeatures_ytarget(sample_n=sample_n)
        h = self.activation(self.Tx(sample_n=sample_n))  # Sigmoid of Tx
        epsilon = 1e-10
        h = np.clip(h, epsilon, 1 - epsilon)
        J = -(y_i * np.log(h) + (1 - y_i) * np.log(1 - h))
        return J 
    def gradient_descent(self,s1):
        x_features,y_i = self.get_xfeatures_ytarget(sample_n=s1)
        if y_i == 1:
            gradient = -(1 - self.activation(self.Tx(sample_n=s1))) * x_features
        if y_i == 0:
            gradient = self.activation(self.Tx(sample_n=s1)) * x_features
        self.gradient = gradient 
        return gradient
    def update_T(self):
        self.theta_vector -= (self.alpha * self.gradient)
    def iterations(self,n=100):
        cost_dict = {}
        iterations = 0 
        for i in range(n):
            total_cost = 0
            iterations +=1
            for index,sample in self.data.iterrows():
                total_cost += self.J(sample_n=index)
                self.gradient_descent(s1=index)
                self.update_T()
            cost_mean = (total_cost / self.m)
            cost_dict[i] = (cost_mean, self.theta_vector.copy())
        return cost_dict
    def stopping_criteria(self, n=100, threshold=0.00001, max_iterations=1000):
        cost_dict = {}
        iteration = 0

        for i in range(n):
            total_cost = 0
            iteration += 1
            for index, sample in self.data.iterrows():
                total_cost += self.J(sample_n=index)
                self.gradient_descent(s1=index)
                self.update_T()

            cost_mean = total_cost / self.m
            cost_dict[i] = (cost_mean, self.theta_vector.copy())

            # Check for stopping criteria
            if iteration > 1:  # We need at least two iterations to calculate a delta
                deltaJ = abs(cost_mean - cost_dict[i - 1][0])  # Change in cost
                if deltaJ < threshold:
                    print(f'Stopping criteria met at iteration {iteration}')
                    break

            if iteration >= max_iterations:
                raise ValueError("Max number of iterations reached")

        print(f'Total iterations completed: {iteration}')
        return cost_dict

        
