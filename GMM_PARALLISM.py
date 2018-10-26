import random
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
import json

from multiprocessing import Pool


class GMM(object):
    def __init__(self, clusters, data, feature_size, epsilon=0.05, max_iterations=200, initializations=100, parallel=True, path='gmm_result/'):
        # random.seed(a=2018)
        self.initializations = initializations
        self.feature_size = feature_size
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.num_of_clusters = clusters
        self.if_parallel = parallel

        self.train_data = data

        self.final_prediction_file_name = path + 'probability_{}.npy'.format(self.num_of_clusters)
        self.model_param_file_name = path + 'best_model_param_{}.json'.format(self.num_of_clusters)

        print("input features: {}".format(len(data)))
        # for uttr in data:
        #     for frame in uttr:
        #         self.train_data.append(frame / np.linalg.norm(frame))

    def initialize(self):

        self.train_data = np.array(self.train_data)  # N by feature size matirx, which is N x 43 in this case.
        models_param = dict(mean=[], weight=[], cov=[])
        for _ in range(self.num_of_clusters):
            sample = random.choice(self.train_data)
            sample_cov = np.array([random.choice(self.train_data) for i in range(200)])
            models_param['mean'].append(sample)
            models_param['weight'].append(1 / self.num_of_clusters)
            models_param['cov'].append(np.cov(sample_cov.T))

        models_param['weight'] = np.array(models_param['weight']).reshape(self.num_of_clusters, -1)
        models_param['mean'] = np.array(models_param['mean'])
        return models_param

    def E_step(self, data, models_param):

        list_of_probability_of_input_given_distribution_true = [multivariate_normal.pdf(data,
                                                                               mean=models_param['mean'][i, :], 
                                                                               cov=models_param['cov'][i])
                                                           for i in range(self.num_of_clusters)]


        list_of_probability_of_input_given_distribution = np.array(list_of_probability_of_input_given_distribution_true)

        list_of_probability_of_input = np.sum(list_of_probability_of_input_given_distribution *
                                           models_param['weight'], axis=0)
        list_of_probability_of_input = list_of_probability_of_input.reshape(1, len(data))

        posteriors = np.divide(
            np.multiply(list_of_probability_of_input_given_distribution, models_param['weight']),
            list_of_probability_of_input)
        return posteriors

    def M_step(self, posteriors, models_param):
        # update thge parameter of models
        posterior_sum = list(np.sum(posteriors, axis=1).reshape(self.num_of_clusters, 1))
        models_param['mean'] = np.array([
            np.divide(
                np.sum(
                    np.multiply(posteriors[i, :].reshape(len(self.train_data), 1), self.train_data),
                    axis=0),
                posterior_sum[i])
                                     for i in range(self.num_of_clusters)])

        models_param['weight'] = np.array([posterior_sum[i] / len(self.train_data) for i in range(self.num_of_clusters)]).reshape(self.num_of_clusters, 1)

        temp = []
        for i in range(self.num_of_clusters):
            weight = posteriors[i, :].reshape(len(self.train_data), 1) # N by 1
            result_first_part = np.matmul(self.train_data.T, self.train_data * weight) / posterior_sum[i]
            mean = models_param['mean'][i, :].reshape(self.feature_size, 1)
            result_second_part = np.matmul(mean, mean.T)
            temp.append(result_first_part - result_second_part)
        models_param['cov'] = temp

        return models_param

    def fixed_cluster_run(self):
        if self.if_parallel:
            with Pool() as p:
                print("Start paralleism")
                result = p.map(self.single_run, list(range(self.initializations)))
        else:
            print("Start sequential...")
            result = [self.single_run() for _ in range(self.initializations)]

        print("Process result...")
        models_param_list = []
        var_list = []
        converge_list = []
        for model_param, var, converged in result:
            models_param_list.append(model_param)
            var_list.append(var)
            converge_list.append(converged)

        index = var_list.index(min(var_list))
        best_data = models_param_list[index]
        converged = converge_list[index]
        
        current_mean_array = np.array(
            [np.linalg.norm(best_data['mean'][i, :]) / self.feature_size for i in range(self.num_of_clusters)])

        current_weight_array = np.array(best_data['weight'])

        current_cov_det_array = np.array(
            [np.linalg.norm(np.diag(best_data['cov'][i])) / self.feature_size for i in range(self.num_of_clusters)])

        print("Best: {}".format(index))
        print(" Mean: {}".format(current_mean_array.tolist()))
        print(" Weight: {}".format(current_weight_array.reshape(self.num_of_clusters,).tolist()))
        print(" Cov: {}".format(current_cov_det_array.tolist()))

        print("if converged: {}".format(converged))

        # self.best = best_data
        self.save_models_param(best_data)

        np.save(self.final_prediction_file_name, self.predict(self.train_data, best_data))

        return current_mean_array.tolist(), \
               current_weight_array.reshape(self.num_of_clusters,).tolist(), \
               current_cov_det_array.tolist()

    def save_models_param(self, models_param):
        models_param_save = models_param.copy()
        models_param_save['mean'] = models_param_save['mean'].tolist()
        models_param_save['weight'] = models_param_save['weight'].tolist()
        models_param_save['cov'] = [cov.tolist() for cov in models_param_save['cov']]
        with open(self.model_param_file_name, 'w') as file:
            json.dump(models_param_save, file)

    def predict(self, data, models_param):
        posteriors = self.E_step(data=data, models_param=models_param)
        probabilities = np.argmax(posteriors, axis=0)
        return probabilities

    def get_average_variance_across_distribution(self, posteriors):
        probabilities = np.argmax(posteriors, axis=0)
        groups = [list() for _ in range(self.num_of_clusters)]

        for probability, value in zip(probabilities, self.train_data):
            groups[probability].append(value.tolist())

        variance = []
        for group in groups:
            matrix = np.array(group).T
            if self.feature_size == 1:
                variance.append(np.var(matrix))
            else:
                variance.append(np.diag(np.cov(matrix)))

        if self.feature_size == 1:
            value = sum(variance) / self.num_of_clusters
        else:
            value = sum(np.linalg.norm(np.array(variance), axis=1)) / self.num_of_clusters

        return value

    def single_run(self, _=None):
        import time
        models_param = self.initialize()
        converged = False

        # current convergence  measure compares difference between the distribution-wise norm
        #   Specifically, for mean, the distribution-wise value is the norm of the mean.
        #                 for cov, the distribution-wise value is the norm of the diagnoal of the covariance matrix.
        #                 for weight, the distribution-wise value is just the number itself.
        #
        #
        #   In future, if you would like to increase the precision of convergence measurement,
        #       consider directly compare raw distribution-wise mean and cov matrix

        for _ in trange(self.max_iterations):
            previous_mean_array = np.array([np.linalg.norm(models_param['mean'][i, :]) for i in range(self.num_of_clusters)])
            previous_weight_array = np.array(models_param['weight'])
            if self.feature_size == 1:
                previous_cov_det_array = np.array(
                    [np.linalg.norm(models_param['cov'][i]) for i in range(self.num_of_clusters)])
            else:
                previous_cov_det_array = np.array([np.linalg.norm(np.diag(models_param['cov'][i])) for i in range(self.num_of_clusters)])

            start = time.time()
            posteriors = self.E_step(data=self.train_data, models_param=models_param)

            print("E_step: {}s".format(time.time() - start))
            start = time.time()
            models_param = self.M_step(posteriors=posteriors, models_param=models_param)

            print("M_step: {}s".format(time.time() - start))

            current_mean_array = np.array([np.linalg.norm(models_param['mean'][i, :]) for i in range(self.num_of_clusters)])
            current_weight_array = np.array(models_param['weight'])

            if self.feature_size == 1:
                current_cov_det_array = np.array([np.linalg.norm(models_param['cov'][i]) for i in range(self.num_of_clusters)])
            else:
                current_cov_det_array = np.array([np.linalg.norm(np.diag(models_param['cov'][i])) for i in range(self.num_of_clusters)])

            mean_difference = current_mean_array - previous_mean_array
            cov_difference = current_cov_det_array - previous_cov_det_array
            weight_difference = current_weight_array - previous_weight_array

            if np.all(abs(mean_difference) < self.epsilon) \
                    and np.all(abs(cov_difference) < self.epsilon) \
                    and np.all(abs(weight_difference) < self.epsilon):
                print("Converged")
                converged = True
                break

        variance = self.get_average_variance_across_distribution(posteriors=posteriors)
        return models_param, variance, converged




def load_data(feature_size, frame_level=True, normalize=False):

    input_train_features = []
    length = []
    for level in ['clean', 'MWC20', 'MWC15', 'MWC10', 'MWC5', 'MWC0']:

        train_dir = 'data/MFCC/{}.npy'.format(level)
        print('looping through level {}'.format(level))
        features = np.load(train_dir)
        if frame_level:
            for uttr in features:
                input_train_features.extend([frame[0:feature_size] for frame in uttr])
            # length.append(len(uttr))
        else:
            input_train_features.extend(features)
        # break

    if normalize:
        temp = input_train_features
        input_train_features = []
        for uttr in temp:
            for frame in uttr:
                input_train_features.append(frame / np.linalg.norm(frame))

    return input_train_features


if __name__ == '__main__':

    FEATURE_SIZE = 21
    # interpret_data(False)
    # load()
    # plot_from_data()
    num_initializations = int(input("Number of initializations? >>> "))
    if_parallel = input("would you like to do it parallel? >>> ").lower().startswith('y')

    start = int(input("num of clusters to start? >>> "))
    end = int(input("num of clusters to end?(exclude) >>> "))

    print('Running {} initializations, do Parallel? {}'.format(num_initializations, if_parallel))

    range_ = list(range(start, end))
    print("looping thourgh {}".format(range_))

    features = load_data(FEATURE_SIZE)


    for i in range_:
        print("Running {} clusters...".format(i))
        model = GMM(i, data=features, feature_size=FEATURE_SIZE, initializations=num_initializations, parallel=if_parallel)
        model.fixed_cluster_run()
    # model.initialize()
    # model.single_run()
    # model.predict()
    # model.plot()
    pass