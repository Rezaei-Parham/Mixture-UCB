import numpy as np
import torch
import os
import cvxpy as cp
import json

import numpy as np
import sklearn.metrics
np.random.seed(0)
def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def compute_reals_nearest_neighbour_distances(real_features,nearest_k):
    return compute_nearest_neighbour_distances(
            real_features, nearest_k)

def compute_linear_term(real_features,fake_features,nearest_k,reals_nnd = None):
    if LINEAR_METRIC == 'kid':
        kernel = KernelUtils.frobenius_norm(fake_features,real_features)
        kernel /= (len(real_features) * len(fake_features))
        return 2 * kernel
    if reals_nnd is None:
        real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
            real_features, nearest_k)
    else:
        real_nearest_neighbour_distances = reals_nnd
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)
    if LINEAR_METRIC == 'precision':
        prec =  (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()
        return prec
    elif LINEAR_METRIC == 'density':
         return (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).sum(axis=0).mean()
    else:
        print("The linear metric is invalid")
        assert 1==0

OGD = False
ALPHA_PRINT = 50
RESCALER = 1
KID_NEW = False
QUADRATIC_METRIC = 'kid'
LINEAR_METRIC = 'kid'
NUMBER_OF_SIMULATIONS = 10
SIGMA = 40
DELTA = 0.03
BETA = 1
INITIAL_SAMPLE_COUNT = 5
OFFLINE_CUTOFF = 3000 
DATASET_REAL_CUTOFF = 3000
MINI_BATCH = 1
DELTA_L = 0.6
DELTA_G = 0.4 
TOTAL_ROUNDS = 3000
BLOCK_SIZE = 800
small_amount = 1e-8
LAMBDA = 1
KNN = 5
feature_extractor = 'dino'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False

class KernelUtils:
    @staticmethod
    def gaussian_kernel(x, y, sigma):
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    @staticmethod
    def frobenius_norm(X, Y, sigma=SIGMA, block_size=BLOCK_SIZE):
        is_renyi = True
        if QUADRATIC_METRIC == 'kid':
            is_renyi = False
        X, Y = torch.Tensor(X).to(DEVICE), torch.Tensor(Y).to(DEVICE)
        n_data_x, n_data_y = X.size(0), Y.size(0)
        sum_frobenius = 0.0

        for i in range(0, n_data_x, block_size):
            for j in range(0, n_data_y, block_size):
                X_block = X[i:i+block_size]
                Y_block = Y[j:j+block_size]
                if is_renyi:
                    kernel_block = KernelUtils.gaussian_kernel(X_block.unsqueeze(0), Y_block.unsqueeze(1), sigma) ** 2
                else:
                    kernel_block = KernelUtils.gaussian_kernel(X_block.unsqueeze(0), Y_block.unsqueeze(1), sigma)
                sum_frobenius += torch.sum(kernel_block).item()

        return sum_frobenius
    
    @staticmethod
    def scaled_kernel(kernel,sizes):
        sizes = 1/np.array(sizes)
        size_matrix = sizes.reshape(sizes.shape[-1],1) @ sizes.reshape(1,sizes.shape[-1])
        if DEBUG:
            print("size matrix",size_matrix.shape)
        kernel_resized = size_matrix * kernel
        if QUADRATIC_METRIC == 'kid' and KID_NEW:
            print(kernel_resized.shape)
            for i in range(len(sizes)):
                kernel_resized[i,i] = kernel[i,i] / (sizes[i] * (sizes[i]-1))
        if DEBUG:
            print("kernel resized size",kernel_resized)
        return kernel_resized

    
class RKEOfflineEvaluator():
    def __init__(self, model_names, dataset_name, has_reference=False):
        self.dataset_name=dataset_name
        self.model_names = model_names
        self.datasets = self._load_all_datasets()
        self.real_dataset = None
        self.use_linear = has_reference
        if has_reference: self._prepare_real_dataset()
        self.number_of_arms = len(model_names)
        self.kernel = np.zeros((self.number_of_arms,self.number_of_arms))
        self.build_kernel_matrix()
        self.optimal_alphas = self.calculate_optimal_alphas(use_linear=has_reference)
        self.optimal_model = self.cal_optimal_model()
        self.true_model_scores = {}
        self.optimal_rke = self.cal_score_with_alphas(self.optimal_alphas,save=True)
        print("offline is done :) -> optimal metric is: ",self.optimal_rke)

    def _prepare_real_dataset(self):
        self.real_dataset = self._load_dataset(self.dataset_name,DATASET_REAL_CUTOFF)
        self.reals_nnd = None
        if LINEAR_METRIC != 'kid':
            self.reals_nnd = compute_reals_nearest_neighbour_distances(self.real_dataset,KNN)
        self.linears = self._compute_linears()

    def _get_dataset_sizes(self):
        return [len(self.datasets[model]) for model in self.model_names]

    def _load_dataset(self, model_name, size=OFFLINE_CUTOFF):
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        if self.dataset_name == 't2t':
            n = np.load(dataset_path)['roberta_features']
        else:
            n = np.load(dataset_path)['dino_features']
        repetition = False
        print("Dataset Size ->",len(n))
        return n[np.random.choice(len(n), size=size, replace=repetition)]

    def _load_all_datasets(self):
        return {model: self._load_dataset(model) for model in self.model_names}

    def _get_dataset_path(self, model_name):
        if self.dataset_name in ['imagenet','ffhq','cifar10','lsun_bedroom','toy','ffhq_truncated','quality_ffhq','afhq_truncated']:
            return os.path.join(f'../{self.dataset_name}/{feature_extractor}/', f'{model_name}.npz')
        elif self.dataset_name == 't2i_cluster':
            return os.path.join('../', f'{model_name}/{cluster_name}.npz')
        elif self.dataset_name == 't2i_coco':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 't2t':
            return os.path.join('../', f'{model_name}_features.npz')
        elif self.dataset_name == 'imagewoof':
            return os.path.join('', f'{model_name}.npz')
        elif self.dataset_name == 'styles':
            return os.path.join('', f'{model_name}.npz')
        elif self.dataset_name == 't2i_dog':
            return os.path.join('', f'{model_name}.npz')
        
    def _compute_linears(self,samples=None):
        linears = {}
        if samples is None:
            for model in self.model_names:
                linears[model] = compute_linear_term(self.real_dataset,self.datasets[model],KNN,self.reals_nnd)
        else:
            for model in self.model_names:
                linears[model] = compute_linear_term(self.real_dataset,samples[model],KNN,self.reals_nnd)
        return linears

    def compute_sample_precision(self,sample):
        if len(sample.shape) == 1:
            sample = sample.reshape(1,sample.shape[0])
        return compute_linear_term(self.real_dataset,sample,KNN,self.reals_nnd)

    def cal_optimal_model(self):
        if QUADRATIC_METRIC == 'rke':
            val = [1/self.kernel[i,i] for i in range(len(self.model_names))]
            if self.use_linear:
                val = [self.kernel[i,i]/(len(self.datasets[self.model_names[i]])**2)-LAMBDA * self.linears[self.model_names[i]] for i in range(len(self.model_names))]
                return np.argmin(val)
            return np.argmax(val)
        elif QUADRATIC_METRIC == 'kid':
            val = [self.kernel[i,i]/(len(self.datasets[self.model_names[i]])**2) - self.linears[self.model_names[i]] for i in range(len(self.model_names))]
            return np.argmin(val)

    def build_kernel_matrix(self):
        for i in range(self.number_of_arms):
            xi = self.datasets[self.model_names[i]]
            for j in range(self.number_of_arms):
                xj = self.datasets[self.model_names[j]]
                self.kernel[i,j] = KernelUtils.frobenius_norm(xi,xj)
        if QUADRATIC_METRIC == 'kid':
            self_kernel = KernelUtils.frobenius_norm(self.real_dataset,self.real_dataset)
            self_kernel /= (len(self.real_dataset) ** 2)
            self.kid_self_kernel = self_kernel

    def cal_score_with_alphas(self,alphas,save=False):
        quadratic = alphas.reshape(1,alphas.shape[-1]) @ KernelUtils.scaled_kernel(self.kernel,self._get_dataset_sizes()) @ alphas.reshape(alphas.shape[-1],1)
        quadratic = quadratic[0,-1]
        if DEBUG:
            print(alphas)
            print("pre, -> ",quadratic)
            print("rke-mc = ",1/quadratic)
        

        if self.use_linear:
            quadratic -= LAMBDA * np.dot(np.array([self.linears[mod] for mod in self.model_names]),alphas)
        if QUADRATIC_METRIC == 'kid':
            quadratic += self.kid_self_kernel
        self.true_model_scores['optimal'] = quadratic
        if not self.use_linear:
            quadratic = 1/quadratic
        return quadratic

    def print_model_scores(self):
        for i in self.model_names:
            datas = self.datasets[i]
            ker = KernelUtils.frobenius_norm(datas[:OFFLINE_CUTOFF],datas[:OFFLINE_CUTOFF],sigma=SIGMA)
            self.true_model_scores[i] = ker
            scaled = ker / (len(datas) ** 2)
            if QUADRATIC_METRIC == 'kid':
                if KID_NEW:
                    ker = ker - len(datas)
                    scaled = ker / (len(datas) * (len(datas) - 1))
                scaled += self.kid_self_kernel
                scaled -= self.linears[i]
                print(scaled)
            elif self.use_linear:
                print("before",scaled)
                scaled = scaled - LAMBDA * self.linears[i]
                print("after",scaled)
            
            if not self.use_linear: print("------>",1/scaled)  

    def calculate_optimal_alphas(self,use_linear=False):
        n_arms = self.number_of_arms
        alphas = cp.Variable(n_arms,nonneg=True)
        kernel = self.kernel
        kernel = KernelUtils.scaled_kernel(kernel,self._get_dataset_sizes()) 
        if DEBUG:
            print('the offline kernel -> ')
            print("kernel ->",kernel)
        kernel_size = kernel.shape[0]
        # the alpha array
        alpha_array = 0
        for j in range(n_arms):
            arm_section = np.zeros(n_arms)
            arm_section[j] = 1
            alpha_array += alphas[j] * arm_section 

        epsilon_eye = np.eye(kernel_size) * 1e-7
        # making sure kernel is psd
        kernel += epsilon_eye
        probabilistic_kernel = cp.quad_form(alpha_array,kernel)

        
        if use_linear:
            linear_array = np.array([self.linears[mod] for mod in self.model_names])
            print("precision array ->",linear_array)
            linear_array = LAMBDA * linear_array
            linear_term = cp.sum(cp.multiply(linear_array, alphas))
            probabilistic_kernel = probabilistic_kernel - linear_term

        objective = cp.Minimize(probabilistic_kernel)
        constraints = [cp.sum(alphas)==1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_alphas = alphas.value
        print("optimal alphas ->",optimal_alphas)
        return optimal_alphas


class RKEOnlineEvaluator():
    def __init__(self, model_names,dataset_name,offline_evaluator,mode='mix_ucb',use_linear=False):
        self.dataset_name = dataset_name
        self.OfflineEvaluator = offline_evaluator
        self.model_names = model_names
        self.mode = mode
        self.model2idx = {}
        self.use_linear = use_linear
        if self.use_linear:
            self.linears = {model_name : [] for model_name in model_names}
        for i in range(len(model_names)):
            self.model2idx[model_names[i]] = i
        self.datasets = self._load_all_datasets()
        self.samples = {model: [] for model in model_names}
        self.sample_indexes = {model: 0 for model in model_names}
        self.number_of_arms = len(model_names)
        self.kernel = np.zeros((len(self.model_names),len(self.model_names)))
        self.scores = []
        

    def _load_dataset(self, model_name):
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        dataset = np.load(dataset_path)[f'{feature_extractor}_features']
        shuffled_indices = np.random.permutation(dataset.shape[0])
        dataset = dataset[shuffled_indices]
        return dataset
    
    def report_sample_nums(self):
        for i in self.model_names:
            print(f"model: {i}, num: {len(self.samples[i])}")

    def _load_all_datasets(self):
        return {model: self._load_dataset(model) for model in self.model_names}

    def _get_dataset_path(self, model_name):
        if self.dataset_name in ['imagenet','ffhq','cifar10','lsun_bedroom','toy','ffhq_truncated','quality_ffhq','afhq_truncated']:
            return os.path.join(f'../{self.dataset_name}/DGM/{feature_extractor}/', f'{model_name}.npz')
        elif self.dataset_name == 't2i_cluster':
            return os.path.join('..', f'{model_name}/{cluster_name}.npz')
        elif self.dataset_name == 't2i_coco':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 't2t':
            return os.path.join('../', f'{model_name}_features.npz')
        elif self.dataset_name == 'imagewoof':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 'styles':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 't2i_dog':
            return os.path.join('..//', f'{model_name}.npz')
        
    def _get_samples(self, model_name, num_samples):
        dataset = self.datasets[model_name]
        if self.sample_indexes[model_name]+num_samples > len(dataset):
            print("ERROR: Too many samples needed")
            assert 0==1
        
        returning_sample = dataset[self.sample_indexes[model_name]:self.sample_indexes[model_name]+num_samples]
        self.sample_indexes[model_name] = self.sample_indexes[model_name] + num_samples
        return returning_sample

    def _extend_samples(self, current_samples, new_samples):
        if len(current_samples)==0:
            return new_samples
        return np.concatenate((current_samples, new_samples), axis=0)


    def initial_alphas(self,round_):
        if self.mode == 'one-arm-oracle':
            model_idx = self.OfflineEvaluator.optimal_model
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        elif self.mode == 'mixture-oracle':
            alphas = self.OfflineEvaluator.optimal_alphas
        elif self.mode in ['mixture-ucb','one-arm-ucb']:
            model_idx = round_ % self.number_of_arms
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        else:
            print('INVALID MODE')
            assert 0==1
        return alphas

    def run_online_evaluation(self, num_rounds, num_samples=MINI_BATCH):
        for round_ in range(0,num_rounds):
            if round_ % 500 == 10:
                print(f"round -- {round_}")
            # score = self._cal_metric_sample_based()
            # self.scores.append(score)
            alphas = self._select_best_model_ucb(round_)
            # alphas /= np.sum(alphas)
            chosen_models = np.random.choice(self.model_names, size=num_samples, p=alphas)
            
            for model_name in chosen_models:
                new_sample = self._get_samples(model_name, 1)  
                self.samples[model_name] = self._extend_samples(self.samples[model_name], new_sample)
                self._update_the_kernel(new_sample,model_name)

            score = self._cal_metric_sample_based()
            self.scores.append(score)

            if round_ % ALPHA_PRINT == 0:
                print("round score -> ",score)
                print("round alphas ->",alphas)
            
            

    def best_rke_model(self,round_):
        if self.mode == 'one-arm-oracle':
            model_idx = self.OfflineEvaluator.optimal_model
        else:
            kernel_matrix = self.kernel
            kernel_matrix = KernelUtils.scaled_kernel(kernel_matrix,self._get_sample_sizes())
            arm_values = [kernel_matrix[i,i] for i in range(len(self.model_names))]
            if self.use_linear:
                precision_array = np.array([np.mean(self.linears[mod]) for mod in self.model_names])
                precision_array = LAMBDA * precision_array
                arm_values -= precision_array
            ucb_values = [DELTA_L * np.sqrt(BETA * np.log(round_) / (2 * len(self.samples[m]) / MINI_BATCH)) for m in self.model_names]
            if round_ % ALPHA_PRINT == 0:
                print('arm val')
                print(arm_values)
                print([len(self.samples[n]) for n in self.model_names])
                print(ucb_values)
                print('done val')
            arm_values_ucb = [arm_values[i]-ucb_values[i] for i in range(len(arm_values))]
            if self.use_linear:
                precision_ucb_values = LAMBDA * np.array([DELTA_G / len(self.samples[m]) * MINI_BATCH for m in self.model_names])
                arm_values_ucb = [arm_values_ucb[i]-precision_ucb_values[i] for i in range(len(arm_values_ucb))]
            model_idx = np.argmin(arm_values_ucb)

        alphas = np.array([0 for _ in range(len(self.model_names))])
        alphas[model_idx] = 1
        return alphas

    def _cal_metric_sample_based(self):
        if QUADRATIC_METRIC == 'rke':
            if not self.use_linear:
                kernel_matrix = self.kernel
                frob = np.sum(kernel_matrix)
                sizes = np.sum(self._get_sample_sizes())
                rke = frob / (sizes * sizes)
                rkemc = 1/rke
                return rkemc
            else:
                kernel_matrix = self.kernel
                frob = np.sum(kernel_matrix)
                sizes = np.sum(self._get_sample_sizes())
                rke = frob / (sizes * sizes)
                linear_sum = 0
                samples = 0
                for i in self.model_names:
                    model_linears = self.linears[i]
                    linear_sum += np.sum(model_linears)
                    samples += len(model_linears)
                linear_sum /= samples
                precision = np.mean(linear_sum)
                return rke - LAMBDA * precision
        elif QUADRATIC_METRIC == 'kid':
            kernel_matrix = self.kernel
            frob = np.sum(kernel_matrix)
            sizes = np.sum(self._get_sample_sizes())
            rke = frob / (sizes * sizes)
            rke += self.OfflineEvaluator.kid_self_kernel
            linear_sum = 0
            samples = 0
            for i in self.model_names:
                model_linears = self.linears[i]
                linear_sum += np.sum(model_linears)
                samples += len(model_linears)
            linear_sum /= samples
            precision = np.mean(linear_sum)
            return rke - LAMBDA * precision

    def _select_best_model_ucb(self,round_):
        if round_ < INITIAL_SAMPLE_COUNT * self.number_of_arms:
            return self.initial_alphas(round_)
        if self.mode in ['one-arm-oracle','one-arm-ucb']:
            return self.best_rke_model(round_)
        if self.mode == 'mixture-oracle':
            return self.OfflineEvaluator.optimal_alphas
        if not OGD:
            is_ucb = True
            alphas = cp.Variable(self.number_of_arms, nonneg=True)
            kernel_matrix = self.kernel
            kernel_matrix = KernelUtils.scaled_kernel(kernel_matrix,self._get_sample_sizes())
            if DEBUG:print(f"kernel size -> {kernel_matrix.shape}")
            epsilon_eye = np.eye(self.number_of_arms) * small_amount
            kernel_matrix += epsilon_eye
            alpha_array = 0
            for j in range(self.number_of_arms):
                arm_section = np.zeros(self.number_of_arms)
                arm_section[j] = 1
                alpha_array += alphas[j] * arm_section
            if DEBUG: print(f"alphas shape -> {alpha_array.shape}")

            quadratic_form = cp.quad_form(alpha_array, kernel_matrix)
            if is_ucb:
                ucb_weights = np.array([DELTA_L * np.sqrt(BETA * np.log(round_) / len(self.samples[m]) * MINI_BATCH) for m in self.model_names])
                linear_term = cp.sum(cp.multiply(ucb_weights, alphas))
                objective = quadratic_form - linear_term
            else:
                objective = quadratic_form

            if self.use_linear:
                precision_array = np.array([np.mean(self.linears[mod]) for mod in self.model_names])
                precision_array = LAMBDA * precision_array
                precision_term = cp.sum(cp.multiply(precision_array, alphas))
                precision_ucb_weights = LAMBDA * np.array([DELTA_G / len(self.samples[m]) * MINI_BATCH for m in self.model_names])
                precision_term += cp.sum(cp.multiply(precision_ucb_weights, alphas))
                objective -= precision_term

            objective = cp.Minimize(objective)
            constraints = [cp.sum(alphas) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            optimal_alphas = alphas.value
        else:
            kernel_matrix = self.kernel
            kernel_matrix = KernelUtils.scaled_kernel(kernel_matrix,self._get_sample_sizes())
            h = kernel_matrix @ (np.array(self._get_sample_sizes())).reshape(len(self._get_sample_sizes()),1)
            h = 2/round_ * h
            ucb_weights = np.array([DELTA_L * np.sqrt(BETA * np.log(round_) / len(self.samples[m]) * MINI_BATCH) for m in self.model_names])
            if self.use_linear:
                precision_array = LAMBDA * np.array([np.mean(self.linears[mod]) for mod in self.model_names])
                precision_ucb_weights = LAMBDA * np.array([DELTA_G / len(self.samples[m]) * MINI_BATCH for m in self.model_names])
                h_wp = [h[i] - precision_array[i] for i in range(len(h))]
                ucb_wp = [ucb_weights[i] - precision_ucb_weights[i] for i in range(len(ucb_weights))]
                h_array = [h_wp[i] - ucb_wp[i] for i in range(len(h))]
            else:
                h_array = [h[i] - ucb_weights[i] for i in range(len(ucb_weights))]
            m = np.argmin(h_array)
            optimal_alphas = [0 for _ in range(len(self.model_names))]
            optimal_alphas[m] = 1

        optimal_alphas /= np.sum(optimal_alphas)
        return optimal_alphas

    def _collect_all_samples(self):
        return np.concatenate([self.samples[model] for model in self.model_names], axis=0)

    # get number of samples from each model
    def _get_sample_sizes(self):
        return [len(self.samples[model]) for model in self.model_names]
    
    def _update_the_kernel(self,samples,samples_model,sigma=SIGMA):
        for model in self.model_names:
            k = KernelUtils.frobenius_norm(samples,self.samples[model])
            self.kernel[self.model2idx[samples_model],self.model2idx[model]] += k
            self.kernel[self.model2idx[model],self.model2idx[samples_model]] += k
            #self.kernel[self.model2idx[samples_model],self.model2idx[samples_model]] -= k
            if self.use_linear:
                prec = self.OfflineEvaluator.compute_sample_precision(samples)
                self.linears[samples_model].append(prec)


    def initiate_kernel_matrix(self):
        for i in range(len(self.model_names)):
            xi = self.samples[self.model_names[i]]
            for j in range(len(self.model_names)):
                xj = self.samples[self.model_names[j]]
                self.kernel[i, j] = KernelUtils.frobenius_norm(xi, xj)
        if self.use_linear:
            for model in self.model_names:
                for sample in self.samples[model]:
                    self.linears[model].append(self.OfflineEvaluator.compute_sample_precision(sample))
        
        


if __name__ == "__main__":
    model_names = ['','',''] # put name of the models to sample from
    dataset_name = 'ffhq_truncated'
    cluster_name = '0.3'
    mode = 'one-arm-ucb'
    experiment = ''
    with_quality = True
    print('MODE is -> ',mode)
    scores_of_simulations = {}
    offline_evaluator = RKEOfflineEvaluator(model_names,dataset_name,has_reference=with_quality)
    offline_evaluator.print_model_scores()
    print(offline_evaluator.optimal_alphas)
    # Put True for offline and False for online experiment
    if True:
        true_scores = offline_evaluator.true_model_scores
        alphas = offline_evaluator.optimal_alphas
    else:
        for simu in range(NUMBER_OF_SIMULATIONS):
            print('SIMULATION ROUND - ',simu)
            online_evaluator = RKEOnlineEvaluator(model_names,dataset_name,offline_evaluator,mode,use_linear=with_quality)
            online_evaluator.run_online_evaluation(num_rounds=TOTAL_ROUNDS)
            online_evaluator.report_sample_nums()
            scores_of_simulations[f'{simu}'] = online_evaluator.scores
        cluster_extension = f"_{cluster_name}" if (cluster_name is not None) else ""
        os.makedirs(f'./{experiment}/{dataset_name}{cluster_extension}/{SIGMA}', exist_ok=True)
        extension = ''
        if mode in ['one-arm-ucb','mixture-ucb']:
            extension += f'-{DELTA_L}'
            if with_quality:
                extension += f'-{DELTA_G}'
        ogd_extension = ""
        if OGD: ogd_extension = "-OGD"
        np.savez(f'./{experiment}/{dataset_name}{cluster_extension}/{SIGMA}/{mode}{extension}{ogd_extension}.npz', **scores_of_simulations)

        offline_evaluator.print_model_scores()
        print(offline_evaluator.optimal_alphas)

