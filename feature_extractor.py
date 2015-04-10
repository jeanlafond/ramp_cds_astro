import numpy as np
import scipy as sp
import sklearn.linear_model as lm

def fourier_regression( Y,X, error, order):
    clf = lm.Lasso(fit_intercept = False, alpha = 10)
    n = X.shape[0]
    print n
    X_reg = np.zeros((n,2 *order + 1))
    for i in range(n):
        for j in range(order):
            X_reg[i,j+1] = sp.sin( j * X[i])/error[i]
            X_reg[i,j+order + 1] = sp.cos( j)/error[i]
    clf.fit(X_reg,Y)
    return clf.coef_

def squared_exponential_periodic_1D(theta, d):
    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    return np.exp(-theta[0] * np.sum(np.sin(abs(d)) ** 2, axis=1))

def fold_time_series(time_point, period, div_period):
    real_period = period / div_period
    return time_point % real_period  # modulo real_period


def unfold_sample(x, color):
    """Operates inplace"""
    real_period = x['period'] / x['div_period']
    phase = (x['time_points_%s' % color] % real_period) / real_period * 2 * np.pi
    order = np.argsort(phase)
    x['phase_%s' % color] = phase[order]
    x['light_points_%s' % color] = np.array(x['light_points_%s' % color])[order]
    x['error_points_%s' % color] = np.array(x['error_points_%s' % color])[order]
    x['time_points_%s' % color] = np.array(x['time_points_%s' % color])[order]

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        order = 20
        num_points_per_period = 200
        bins_per_period = 10
        sampling_rate = num_points_per_period / bins_per_period
        t_test = np.linspace(-2 * np.pi, 4 * np.pi, 3 * num_points_per_period)
        num_gp_bins = 10
        gp_bins = [i * 2 * np.pi / num_gp_bins for i in range(num_gp_bins + 1)]
   
        X = []
        ii = 0
        for x in X_dict:
            ii += 1
            if ii / 100 * 100 == ii:
                print ii
            real_period = x['period'] / x['div_period']
            x_new = [x['magnitude_b'], x['magnitude_r'], real_period, x['asym_b'], x['asym_r']]
            for color in ['r', 'b']:
                unfold_sample(x, color=color)
                x_train = x['phase_' + color]
                y_train = x['light_points_' + color]
                y_sigma = x['error_points_' + color]
                res = fourier_regression(y_train,x_train,y_sigma,order)
                for coef in res: x_new.append( coef)
            X.append(x_new) 
        return np.array(X) 
