import numpy as np
import GMM_PARALLISM

from sklearn.preprocessing import normalize
"""
Below is parameter to generate random normal distributed data and run GMM against it
feel free to change the parameter to play around.
"""

clusters = 10
generate_use_mean = np.linspace(0,50,clusters)
generate_use_sigma = [5] * clusters
data = 1000
dimensions = 1

def genrate_data():
    samples = []
    for i in range(clusters):
        s = np.random.normal(generate_use_mean[i], [generate_use_sigma[i]] * dimensions, (data, dimensions))
        samples.extend(s)
    np.save("temp", samples)

def test():
    samples = np.load("temp.npy")
    model = GMM_PARALLISM.GMM(clusters=clusters, data=samples, feature_size=dimensions, max_iterations=300, initializations=15, epsilon=1e-5)
    # model.single_run()
    return model.fixed_cluster_run()



def plot_from_data(mean, weight, variance,if_normalized=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import math

    mu = mean
    assert len(mean) == len(variance) == len(weight)
    # mu = [12.598925102440496, 22.58665775977064, 20.20308653261069, 32.40876068534021, 19.97740990554982, 24.974410048121822]

    # if variance == None:
    # variance = [7.018256494079792, 50.0552250761815, 0.23481541876822123, 6.941408880878499, 0.2110042479254162, 0.2369718004632659]


    for i in range(len(mu)):   # +1 is adding the total variance of all data
        mu_ = mu[i]
        variance_ = variance[i]
        sigma = math.sqrt(variance_)
        x = np.linspace(mu_ - 3 * sigma, mu_ + 3 * sigma, 100)
        plt.plot(x, mlab.normpdf(x, mu_, sigma) * weight[i])
        # plt.show()
        # print("wait")
    if if_normalized:
        samples = np.load("temp.npy")
        zero_mean_samples = samples - samples.mean()
        normalized = normalize(zero_mean_samples, norm='max', axis=0)
        plt.hist(normalized, 200, density=True)
    else:
        samples = np.load("temp.npy")
        obtain_norm = np.linalg.norm(samples, axis=1) / dimensions
        plt.hist(obtain_norm, 200, density=True)
    # plt.scatter(samples)
    plt.show()


# def sklearn_result():
#     desired_clusters = 15
#     model = BayesianGaussianMixture(n_components=desired_clusters, init_params='kmeans', max_iter=200, n_init=10)
#     # samples = np.load("temp.npy")
#     samples = GMM_PARALLISM.load_data(feature_size=21)
#
#     model.fit(samples)
#     np.save('gmm_result/probability_sklearn.npy', model.predict(samples))
#     print(model.weights_)
#     # plot_from_data(model.means_, model.covariances_.reshape(desired_clusters,), model.weights_)
#     # print("sklearn_result")
#     # print("Mean: {}".format(model.means_))
#     # print("Cov: {}".format(model.covariances_.reshape(clusters,)))
#     # print("done")

if __name__ == '__main__':

    genrate_data()
    mean, weight, cov = test()
    plot_from_data(mean, weight, cov, if_normalized=False)
# plot_from_data([114.40401017576025, 115.82880710191789, 125.79893553941183, 120.73827401152765, 117.65187338830759, 57.22687686904379],
#                [0.1636969864432147, 0.4101053986754992, 0.18144976367218837, 0.08885740160568066, 0.1154176415548314,
#                 0.040472808048585225],
#                np.array([1.9063708612219815e+41, 5.260267669740528e+34, 2.8655393227794405e+47, 5.758991043305487e+48,
#                 1.33668865409969e+40, 1.567449135390887e+33]) / 1e+33, if_normalized=False)
#     sklearn_result()


# Mean: [0.0005252731656706822, 0.00047549229117929833]
# Weight: [0.4751286007373968, 0.5248713992626032]
# Cov: [0.37520541785403, 0.3751046134454572]


