import matplotlib.pyplot as plt
import numpy as np
np.random.seed(30)

class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """
    def __init__(self, mu, sigma, g, g_jac, h, h_jac, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = g
        self.g_jac = g_jac
        self.R = R
        # measurement model
        self.h = h
        self.h_jac = h_jac
        self.Q = Q
        alpha_init = 2
        self.alpha = alpha_init
        mu_x_init = 1
        self.mu_x = mu_x_init
        x_init = 2
        self.x = x_init
        

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
        z = (np.sqrt((self.alpha*sensor_data)**2 + 1) + np.random.normal(0,1,1)).reshape(-1,1)
        #z = sensor_data.reshape(-1,1)
        #print(np.shape(z))
        self._predict()
        mu, covariance,alpha = self._update(z)

        return mu,covariance, z

    def _predict(self):
        # FILL in your code here
        self.mu_bar = self.g 
        #print(np.shape(self.mu_bar))
        self.sigma_bar = np.matmul(np.matmul(self.g_jac.astype(float),self.sigma.astype(float)),self.g_jac.T.astype(float)) + self.R
        #print(np.shape(self.sigma_bar))

        matrix = np.matmul(np.matmul(self.h_jac.astype(float),self.sigma_bar),self.h_jac.T.astype(float)) + self.Q
        #print(np.shape(matrix))
        inv_matrix = np.linalg.inv(matrix)
        self.KalmanGain = np.dot(np.matmul(self.sigma_bar,self.h_jac.T.astype(float)),inv_matrix)
        #print(np.shape(self.KalmanGain))



    def _update(self, z):
        # FILL in your code here
        diff_matrix = z - self.h
        self.mu = self.mu_bar + np.dot(self.KalmanGain,diff_matrix)
        self.sigma = self.sigma_bar - np.matmul(np.matmul(self.KalmanGain,self.h_jac.astype(float)),self.sigma_bar) 
        #print(np.shape(self.sigma_bar))
        #print(np.shape(self.mu))
        '''
        ##Updating x, mu_x, alpha, g, g_jac,h, h_jac
        '''
        self.mu_x = self.mu_x*self.alpha
        self.x = self.alpha*self.x + np.random.normal(0,1,1)
        self.alpha = self.mu[1]
        self.g = np.array([self.alpha*self.mu_x,self.alpha]).reshape(-1,1)
        self.g_jac = np.array([[self.alpha,self.x],[0,1]])
        self.h  = np.sqrt((self.alpha*self.mu_x)**2 + 1)
        self.h_jac = np.array([self.x/np.sqrt(self.x**2 + 1),0]).reshape(-1,1).T

        return self.mu, self.sigma, self.alpha

def plot_prediction(t, ground_truth, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    plt.fill_between(
        t,
        pred_x-pred_x_std,
        pred_x+pred_x_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(
        t,
        pred_a-pred_a_std,
        pred_a+pred_a_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")

    plt.show()


def problem3():
    # FILL in your code here
    mu_init = np.array([1,2]).reshape(-1,1)
    sigma_init = np.array([[2,0],[0,2]])
    mu = mu_init
    sigma = sigma_init
    R = np.array([[0.5,0],[0,0.5]])
    #Q = np.array([[1]])
    Q = 1
    x_init = 2
    x = x_init
    alpha_init = 2
    alpha = alpha_init
    mu_x_init = 1
    mu_x = mu_x_init
    g = np.array([alpha*mu_x,alpha]).reshape(-1,1)
    state = np.array([alpha*x,alpha]).reshape(-1,1)
    g_jac = np.array([[alpha,x],[0,1]])
    h  = np.sqrt((alpha*mu_x)**2 + 1)
    h_jac = np.array([x/np.sqrt(x**2 + 1),0]).reshape(-1,1).T

    EKF = ExtendedKalmanFilter(mu,sigma,g,g_jac,h,h_jac,R,Q)
    
    ground_truth = np.ndarray(shape=(20,2))
    measurement = np.ndarray(shape=(20,1))
    Final_predict_mu = np.ndarray(shape=(20,2))
    Final_predict_cov = np.ndarray(shape=(20,2,2))
    time_steps = np.arange(0,20,1)
    
    for t in range(20):#Assuming time-step to be 1 sec 
        if t == 0:
            ground_truth[0,:] = 2
            sensor_data = np.sqrt((alpha*x)**2+1)
            Final_predict_mu[t,:] = mu_init.reshape(2,)
            Final_predict_cov[t,:] = sigma_init
        else:
            ground_truth[t,:] = x #+ np.random.normal(0,1,1)
            #print(np.shape(ground_truth[:,1]))
            sensor_data = np.sqrt((alpha*x)**2+1) #+ np.random.normal(0,1)
            '''
            if t ==0:
                sensor_data = 2
            else:
                sensor_data = ground_truth[t-1,1]
                #print(sensor_data)
            '''
            mu, covariance, z = EKF.run(sensor_data)
            measurement[t,:] = z
            mu = mu.reshape(2,)
            Final_predict_mu[t,:] = mu
            Final_predict_cov[t,:,:] = covariance
            x = 0.1*x


    plot_prediction(time_steps,ground_truth,Final_predict_mu,Final_predict_cov)
    




if __name__ == '__main__':
    problem3()
