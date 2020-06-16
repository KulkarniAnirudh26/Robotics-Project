import matplotlib.pyplot as plt
import numpy as np
np.random.seed(30) #For consistent results.  


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param A: process model
        :param C: measurement model
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.A = A
        self.R = R
        # measurement model
        self.C = C
        self.Q = Q



    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each update/predict cycle, over T timesteps.
        """
        # FILL in your code here
        s = np.array([sensor_data,0,0,0]).reshape(-1,1)
        z = np.matmul(self.C.astype(float),s.astype(float)) + np.random.normal(0,1,1)
        #print(np.shape(z))
        self._predict()
        mu, covariance = self._update(z)


        return mu, covariance, z
        
    

    def _predict(self):
        # FILL in your code here
        #self.mu = self.mu.reshape(-1,1)
        #print(np.shape(self.A))
        #print(np.shape(self.mu))
        self.mu_bar = np.matmul(self.A,self.mu)
        #print(np.shape(self.mu_bar))
        self.covariance_bar = np.matmul(np.matmul(self.A,self.sigma),self.A.T) + self.R
        #print(np.shape(self.covariance_bar))
        matrix = np.matmul(np.matmul(self.C,self.covariance_bar),self.C.T) + self.Q
        #print(np.shape(matrix))
        #if np.size(matrix) != 1:
        inv_matrix = np.linalg.inv(matrix)
        #else:
        #    inv_matrix = np.reciprocal(matrix)
        
        self.Kalman_gain = np.matmul(np.matmul(self.covariance_bar,self.C.T),inv_matrix)

        #return self.mu_bar, self.covariance_bar

    def _update(self, z):
        # FILL in your code here
        diff_matrix = z - np.matmul(self.C,self.mu_bar)
        self.mu = self.mu_bar + np.dot(self.Kalman_gain,diff_matrix)
        self.sigma = self.covariance_bar - np.matmul(np.matmul(self.Kalman_gain,self.C),self.covariance_bar)

        return self.mu, self.sigma

        #print(np.shape(self.sigma))
        #print(np.shape(self.mu))

        



def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """

    #print(np.shape(t))
    #print(np.shape(ground_truth))
    #print(np.shape(measurement))
    #print(np.shape(predict_mean))
    #print(np.shape(predict_cov))

    predict_pos_mean = predict_mean[:, 0]
    predict_pos_std = predict_cov[:, 0, 0]
    

    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    plt.fill_between(
        t,
        predict_pos_mean-predict_pos_std,
        predict_pos_mean+predict_pos_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    plt.show()


def plot_mse(t, ground_truth, predict_means):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors, axis=0) ** 2
    print(mse[99])

    plt.figure()
    plt.plot(t, mse)
    plt.xlabel("time (s)")
    plt.ylabel("position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    plt.show()


def problem2a():
    # FILL in your code here
    
    mu_init = np.array([5,1,0,0]).reshape(-1,1)
    sigma_init = np.array([[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
    mu = mu_init
    sigma = sigma_init
    # process model
    A = np.array([[1,0.1,0,0],[0,1,0.1,0],[0,0,1,0.1],[0,0,0,1]])
    R = 0
    # measurement model
    C = np.array([1,0,0,0]).reshape(-1,1).T
    Q = np.array([[1]])
    KF = KalmanFilter(mu,sigma,A,C,R,Q); 

    
    predict_Mean_over_N_timesteps = np.ndarray(shape = (10000,100,4))
    Covariance_over_N_timesteps = []


    
    for n in range(10000):
        ground_truth = np.ndarray(shape=(100,1))#P(t)
        measurement = np.ndarray(shape=(100,1)) #Z(t)
        sensor_data = np.ndarray(shape=(100,1)) #~P(t)
        Final_mean = []
        Final_covariance = []
        Final_predict_mu = np.ndarray(shape=(100,4))
        Final_predict_cov = np.ndarray(shape=(100,4,4))
        time_steps = np.arange(0,100,1)
        KF.reset()
        for t in range(len(time_steps)):#100 time steps
            '''
            if t == 0:
                a = np.sin(0.1*time_steps[t])
                ground_truth[t,:] = a
                sensor_data[t,:] = a + np.random.normal(0,1,1)
                s = np.array([sensor_data[t,:],0,0,0]).reshape(-1,1)
                z = np.matmul(C.astype(float),s.astype(float)) + np.random.normal(0,1,1)
                measurement[t,:] = z
                predict_mu_bar = mu_init.reshape(4,)
                Final_predict_mu[t,:] = predict_mu_bar
                Final_predict_cov[t,:,:] = sigma_init
            else:
               ''' 
            a = np.sin(0.1*time_steps[t])
            ground_truth[t,:] = a
            sensor_data[t,:] = a + np.random.normal(0,1,1)
            #print(sensor_data[t,:])
            #print(np.shape(ground_truth))
            #print(np.shape(sensor_data))
            predict_mu_bar, predict_covar_bar, z = KF.run(sensor_data[t,:])
            #print(np.shape(z))
            measurement[t,:] = z
            predict_mu_bar = predict_mu_bar.reshape(4,)
            Final_predict_mu[t,:] = predict_mu_bar
            Final_predict_cov[t,:,:] = predict_covar_bar
        predict_Mean_over_N_timesteps[n,:,:]  = Final_predict_mu
        #Covariance_over_N_timesteps.append(Final_covariance)

    plot_prediction(time_steps, ground_truth, measurement, Final_predict_mu,Final_predict_cov)
    plot_mse(time_steps, ground_truth,predict_Mean_over_N_timesteps)
    #plot_mse(np.array(time_steps),)

    # Mesaaure error
    # Plot

def problem2b():
    # FILL in your code here
    mu_init = np.array([5,1,0,0]).reshape(-1,1)
    sigma_init = np.array([[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
    mu = mu_init
    sigma = sigma_init
    # process model
    A = np.array([[1,0.1,0,0],[0,1,0.1,0],[0,0,1,0.1],[0,0,0,1]])
    R = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
    # measurement model
    C = np.array([1,0,0,0]).reshape(-1,1).T
    Q = Q = np.array([[1]])
    KF = KalmanFilter(mu,sigma,A,C,R,Q); 

    
    predict_Mean_over_N_timesteps = np.ndarray(shape = (10000,100,4))
    Covariance_over_N_timesteps = []

    
    for n in range(10000):
        ground_truth = np.ndarray(shape=(100,1))#P(t)
        measurement = np.ndarray(shape=(100,1)) #Z(t)
        sensor_data = np.ndarray(shape=(100,1)) #~P(t)
        Final_mean = []
        Final_covariance = []
        Final_predict_mu = np.ndarray(shape=(100,4))
        Final_predict_cov = np.ndarray(shape=(100,4,4))
        time_steps = np.arange(0,100,1)
        KF.reset()
        for t in range(len(time_steps)):#100 time steps
            '''
            if t == 0:
                a = np.sin(0.1*time_steps[t])
                ground_truth[t,:] = a
                sensor_data[t,:] = a + np.random.normal(0,1,1)
                s = np.array([sensor_data[t,:],0,0,0]).reshape(-1,1)
                z = np.matmul(C.astype(float),s.astype(float)) + np.random.normal(0,1,1)
                measurement[t,:] = z
                predict_mu_bar = mu_init.reshape(4,)
                Final_predict_mu[t,:] = predict_mu_bar
                Final_predict_cov[t,:,:] = sigma_init
            else:
                '''
            a = np.sin(0.1*time_steps[t])
            ground_truth[t,:] = a
            sensor_data[t,:] = a + np.random.normal(0,1,1)
            #print(sensor_data[t,:])
            #print(np.shape(ground_truth))
            #print(np.shape(sensor_data))
            predict_mu_bar, predict_covar_bar, z = KF.run(sensor_data[t,:])
            #print(np.shape(z))
            measurement[t,:] = z
            predict_mu_bar = predict_mu_bar.reshape(4,)
            Final_predict_mu[t,:] = predict_mu_bar
            Final_predict_cov[t,:,:] = predict_covar_bar
        predict_Mean_over_N_timesteps[n,:,:]  = Final_predict_mu
        #Covariance_over_N_timesteps.append(Final_covariance)

    plot_prediction(time_steps,ground_truth,measurement,Final_predict_mu,Final_predict_cov)
    plot_mse(time_steps, ground_truth,predict_Mean_over_N_timesteps)
    

if __name__ == '__main__':
    #print('I am in main')
    problem2a()
    problem2b()