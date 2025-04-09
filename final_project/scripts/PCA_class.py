
class PCA:
    def __init__(self,data,features_as_columns=True,class_label=None):  
        import pandas as pd
        self.data = data.copy()
        self.class_label = class_label
        self.eigen_data = None
        self.eigenvalues = None
        self.eigvectors = None 
        if class_label:
            self.class_label_value = self.data.loc[:,f'{class_label}']
            self.data.drop(columns=class_label,inplace=True)  
    def mean_centering(self,data):
        return self.data.apply(lambda x: x-x.mean())    
    def covariance_matrix(self,data):
        return data.cov()
    def decomposition(self,data):
        import numpy as np
        eigenvalues,eigenvectors = np.linalg.eig(data)
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:,sorted_index]
        return sorted_eigenvalues,sorted_eigenvectors
    def do_PCA(self,n_components=2):
        normalized_data  = self.mean_centering(self.data)
        self.cov_matrix = self.covariance_matrix(normalized_data)
        self.eigenvalues,self.eigenvectors = self.decomposition(self.cov_matrix)
        eigen_data = normalized_data @ self.eigenvectors[:,:n_components]
        self.eigen_data = eigen_data
        return eigen_data
    def PCplot(self):
        import seaborn as sns
        if self.class_label:
            self.eigen_data["target"] = self.class_label_value
            ax = sns.scatterplot(self.eigen_data,x=self.eigen_data[0],y=self.eigen_data[1],hue=self.eigen_data['target'])
            ax.set(xlabel=f'PC1: {round(self.eigenvalues[0] / sum(self.eigenvalues) * 100, 2)}%',ylabel=f'PC2: {round(self.eigenvalues[1] / sum(self.eigenvalues) * 100, 2)}%')

        else:
            ax = sns.scatterplot(data=self.eigen_data,x=self.eigen_data[0],y=self.eigen_data[1])
            ax.set(xlabel=f'PC1: {np.round(self.eigenvalues[0] / sum(self.eigenvalues) * 100, 2)}%',ylabel=f'PC2: {np.round(self.eigenvalues[1] / sum(self.eigenvalues) * 100, 2)}%')

    def scores(self):
        import numpy as np
        percent_variance = [(x/sum(self.eigenvalues))*100 for x in self.eigenvalues]
        percent_variance = [np.round(x,2) for x in percent_variance]
        self.percent_variance_explained = percent_variance
        print(f"Total Variance:{sum(self.eigenvalues)}")
        print("Variance,Percent Variance")
        print([(i,j) for i,j in zip(self.eigenvalues,percent_variance)])
        loadings = self.eigenvectors * np.sqrt(self.eigenvalues)
        self.loadings = loadings
        print("Loadings")
        print(loadings)

        

        
        

