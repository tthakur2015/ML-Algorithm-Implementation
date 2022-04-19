import numpy as np

class GDA:

    # Class attributes

    # GDA instance attributes
    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

        # Instance attribute class count
        self.ones_count = 0
        self.zeros_count = 0 

        # Instance attribute class features
        ones_features = []
        zeros_features = []
    
        # Counting and separating class labels and corresponding feature data
        for data, label in zip(train_data, train_label):
            if label == 1:
                self.ones_count += 1
                ones_features.append(list(data))
            else:
                self.zeros_count += 1
                zeros_features.append(list(data))
        
        num_examples, num_features = np.shape(self.train_data)

        # Calculating class priors
        self.ones_prior = self.ones_count/num_examples
        self.zeros_prior = 1 - self.ones_prior

        # Calculating feature means
        ones_arr = np.array(ones_features)
        zeros_arr = np.array(zeros_features)

        self.ones_mean = np.mean(ones_arr, 0)
        self.zeros_mean = np.mean(zeros_arr, 0)

        # Constructing a covariance matrix 
        ones_delta = ones_features - self.ones_mean
        zeros_delta = zeros_features - self.zeros_mean

        # Covariance matrix of a n dimensional feature vector will be a nxn matrix
        # Cov[x] = E[(x-E(x))(x-E(x))T]
        # Assume that the covariance for both classes is the same
        self.covMatrix = []
        for i in ones_delta:
            i = i.reshape(1,num_features)
            self.covMatrix.append(i.T.dot(i))
        for j in zeros_delta:
            i = i.reshape(1,num_features)
            self.covMatrix.append(i.T.dot(i))

        self.covMatrix = np.array(self.covMatrix)
        self.covMatrix = np.sum(self.covMatrix,0)/num_examples
        self.ones_mean = np.reshape(self.ones_mean, num_features)
        self.zeros_mean = np.reshape(self.zeros_mean, num_features)
    
    def multi_gaussian_prob_density(self, x, mean, cov):
        dim = np.shape(cov)[0]
        # Cov measures of the determinant is zero
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # Probability Density
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob
    
    def predict(self, test_data):
        predict_label = []
        for data in test_data:
            positive_pro = self.multi_gaussian_prob_density(data,self.ones_mean,self.covMatrix)
            negetive_pro = self.multi_gaussian_prob_density(data,self.ones_mean,self.covMatrix)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label



features = [[1,2],[3,4], [5,5], [2,4]]
labels = [0,0,1,1]

y = GDA(features,labels)
y.predict(np.array([[3,4]]))