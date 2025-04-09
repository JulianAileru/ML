class ANN:
    def __init__(self,data,target="target",bias=1,n_layers=3,hidden_units=2,output_units=2):
        import numpy as np 
        self.bias = bias 
        self.target_name = target
        self.n_layers = n_layers #3 layers 
        self.hidden_units = hidden_units #2 hidden units
        self.output_units = output_units #2 output units

        self.data = data
        self.target = self.data[self.target_name].to_numpy()
        self.data = self.data.drop(columns=self.target_name)
        self.n_features = len(self.data.columns)
        self.features = self.data.columns.to_list()
    
        self.data = self.data.to_numpy()
        self.n_samples = self.data.shape[0]
        #self.Input_W = self.weight_initialization()
        #self.Hidden_W = self.weight_initialization(input_layer=False)
        self.Input_W = self.golorot_initialization(input_layer=True)
        self.Hidden_W = self.golorot_initialization(input_layer=False)
        self.Input_B = np.zeros(self.hidden_units)  # One bias for each hidden unit
        self.Hidden_B = np.zeros(self.output_units)  # One bias for each output unit

    def weight_initialization(self,input_layer=True): #initalize weights and add bias 
        import numpy as np 
        if input_layer:
            theta = np.random.rand(self.n_features,self.hidden_units)
            theta = np.insert(theta,0,[self.bias,self.bias],axis=0) #insert bias into both neurons 
            return theta
        else:
            phi = np.random.rand(self.hidden_units,self.output_units)
            phi = np.insert(phi,0,[self.bias,self.bias],axis=0)
            return phi
    def golorot_initialization(self,input_layer=True): #intialization of weights, using Xavier/Golorot method: ideal for sigmoid activation 
        import numpy as np 
        if input_layer:
            # initializing weights for input layer to first hidden layer
            output_units = self.hidden_units
            x = np.sqrt(6/(self.n_features+output_units))
            weights = np.random.uniform(-x,x,size = (self.n_features+1,output_units))  
        else:
            # initializing weights for hidden layer to output layer
            input_units = self.hidden_units
            x = np.sqrt(6/(input_units+self.output_units))
            weights = np.random.uniform(-x,x,size = (input_units+1,self.output_units)) #add bias weight, randomly initalized
        return weights 
    def sigmoid_activation(self,z):
        import numpy as np 
        activity = 1 / (1 + np.exp(-z))
        return activity
    def calculate_z(self,W,x):
        import numpy as np 
        linear_combination = np.dot(W,x)
        return linear_combination 
    def forward_propagation(self,sample_n=0): #FP computed for each sample
        import numpy as np 
        datawbias = np.insert(self.data[sample_n,:],0,self.bias)
        a1 = self.sigmoid_activation(self.calculate_z(self.Input_W.T,datawbias))
        a1 = np.insert(a1,0,1,axis=0) #add bias into activity vector 
        self.a1 = a1 
        a2 = self.sigmoid_activation(self.calculate_z(self.Hidden_W.T,a1))
        self.output_activity = a2 
        return a2
    def compute_cost(self,sample=0,classification_b=True): #J computed for each sample
        import numpy as np 
        if classification_b:
            a2 = self.forward_propagation(sample_n=sample)
            y_i = self.target[sample]
            cross_entropy = - (y_i * np.log(a2) + (1 - y_i) * np.log(1 - a2))
            return cross_entropy
    def back_propagation(self,sample=0,alpha=0.001): #computed after FP of each sample, process of simulatenous update of parameters
        import numpy as np 
        self.Input_B = self.Input_W[0] #put bias parameter in own vector 
        self.Hidden_B = self.Hidden_W[0]
        self.Hidden_W = self.Hidden_W[1:,:] #remove bias from weight vector
        self.Input_W = self.Input_W[1:,:] #remove bias from weight vector
        target = self.target[sample]
        delta3 = self.output_activity - target #a2 - y shape (2,)

        delta2_W = delta3.dot(self.Hidden_W.T) * (self.a1[1:]*(1-self.a1[1:]))  #dot product of delta3 and weights in hidden layer (2X1).T* (2x2)
        delta2_B = delta3.dot(self.Hidden_B) * (self.a1[1] * (1 - self.a1[1])) 

        #update bias
        self.Input_B -= alpha * delta2_B
        #update weights 
        self.Hidden_W -= alpha * np.outer(self.a1[1:], delta3)  # Use a1[1:] to exclude bias
        self.Input_W -= alpha * np.outer(self.data[sample], delta2_W)  # Update input weights with outer product

        #add bias back into weights vector for subsequent iterations
        self.Input_W = np.vstack((self.Input_B.reshape(1, -1), self.Input_W))
        self.Hidden_W = np.vstack((self.Hidden_B.reshape(1,-1),self.Hidden_W))
    def stopping_criteria(self, n=100, threshold=0.00001, max_iterations=100000):
        stop = False
        n_iterations = 0
        iter_cost = {}
        while not stop:
            computed_cost = 0
            n_iterations += 1
            for i in range(self.n_samples):
                self.forward_propagation(sample_n=i)
                computed_cost += self.compute_cost(sample=i)
                self.back_propagation(sample=i)
            
            # Record the average cost per iteration
            iter_cost[n_iterations] = computed_cost / self.n_samples
            
            # Check stopping criteria after the first iteration
            if n_iterations > 1:
                deltaJ = abs(iter_cost[n_iterations - 1] - iter_cost[n_iterations])
                # Ensure both outputs meet the threshold
                if deltaJ[0] < threshold and deltaJ[1] < threshold:
                    print(f'Stopping criteria met at iteration {n_iterations}')
                    stop = True
            if n_iterations >= max_iterations:
                print("maximum number of iterations reached")
                break
                #raise ValueError("Maximum Number of Iterations Reached")
        return iter_cost
    def iterations(self,n_iterations):
        iterations = 0 
        iter_cost = {}
        for i in range(n_iterations):
            computed_cost = 0 
            iterations += 1
            for j in range(self.n_samples):
                self.forward_propagation(sample_n=j)
                computed_cost += self.compute_cost(sample=j) #accumulate cost for each sample
                self.back_propagation(sample=j)
            iter_cost[iterations] = (computed_cost/self.n_samples) #average cost per iteration
        return iter_cost

                



       
    

