from __future__ import division
from bandit_utils import Bandit, mvn_scan, compute_info_gain
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import theano.tensor.slinalg

class Model(object):
    @staticmethod
    def build(N, D, K, mnp, masknp, w_init,r_init, cov_data_name, w_data_name, posterior_method):

        if posterior_method == 'sgld':
            return Model._sgld(N, D, K, mnp, masknp, w_init,r_init)
        elif posterior_method == 'svi':
            return Model._svi(N, D, K, mnp, masknp, w_init,r_init)
        elif posterior_method == 'empirical':
            return Model._empirical(N,D,K,mnp,masknp)
        elif posterior_method == 'low_rank_emp':
            return Model._lowrnk_emp(N,D,K,mnp,masknp)
        elif posterior_method == 'true':
            return Model._true(N,D,K,mnp,masknp,cov_data_name, w_data_name)
        else:
            raise Exception(posterior_method + 'posterior estimation method not implemented.')


    @staticmethod
    def build_updates(N, D, K, m_params, v_params, y, cov, mask, z_eps, posterior_method, gd_limit=500):

        if posterior_method == 'sgld':
            return Model._sgld_updt(N, D, K, m_params, v_params, y, cov, mask, z_eps, gd_limit)
        if posterior_method == 'svi':
            return Model._svi_updt(N, D, K, m_params, v_params, y, cov, mask, z_eps, gd_limit)
        else:
            raise Exception(posterior_method + 'posterior estimation method not implemented.')

    @staticmethod
    def _sgld(N, D, K, mnp, masknp, w_init, r_init):
        Shared = lambda shape, name: theano.shared(value=np.ones(shape, dtype=theano.config.floatX),
                                                   name=name, borrow=True)
        srng = T.shared_randomstreams.RandomStreams(seed=120)

        mask = Shared((D, N), 'mask')
        mask.set_value(masknp)
        m = T.as_tensor_variable(mnp)
        y = mask * m
        zero_y = T.as_tensor_variable(np.zeros((D, N)))
        zero2 = T.as_tensor_variable(np.zeros((D, D)))
        zero = T.as_tensor_variable(np.zeros(D))
        st = T.sum(T.neq(y, zero_y), axis=0)
        s = st.eval()

        # Define model parameters
        w = Shared((D, K), 'w')
        w.set_value(w_init)
        r = Shared((K), 'r')
        r.set_value(r_init)
        gamma = Shared((1), 'gamma')
        gamma0 = Shared((1), 'gamma0')
        c0 = Shared((1), 'c0')
        sigma = Shared((1), 'sigma')

        # Define random variables for mVNscan component
        z_y = srng.normal([D])
        z_k = srng.normal([K])
        z_eps = srng.normal()

        # For data given seqentially we need a different covariance matrix for each yn
        wwT = T.dot(w, w.T)
        cov = Shared((D, D), 'cov')
        cov = wwT + sigma[0] * T.identity_like(wwT)
        # cov = T.as_tensor_variable(covnp, name= 'cov')

        return mask, m, y, zero_y, zero2, zero, st, w, r, gamma, gamma0, c0, sigma, z_y, z_k, z_eps, wwT, cov

    @staticmethod
    def _svi(N, D, K, mnp, masknp, w_init, r_init):
        Shared = lambda shape, name: theano.shared(value=np.ones(shape, dtype=theano.config.floatX),
                                                   name=name, borrow=True)
        srng = T.shared_randomstreams.RandomStreams(seed=120)

        mask = Shared((D, N), 'mask')
        mask.set_value(masknp)
        m = T.as_tensor_variable(mnp)
        y = mask * m
        zero_y = T.as_tensor_variable(np.zeros((D, N)))
        zero2 = T.as_tensor_variable(np.zeros((D, D)))
        zero = T.as_tensor_variable(np.zeros(D))
        st = T.sum(T.neq(y, zero_y), axis=0)
        s = st.eval()

        # Define variational parameters
        m_w = Shared((D, K), 'm_w')
        m_w.set_value(w_init)
        s_w = Shared((D, K), "s_w")
        m_r = Shared((K), 'm_r')
        m_r.set_value(r_init)
        s_r = Shared((K), 's_r')
        m_gamma = Shared((1), 'm_gamma')
        s_gamma = Shared((1), 's_gamma')
        m_gamma0 = Shared((1), 'm_gamma0')
        s_gamma0 = Shared((1), 's_gamma0')
        m_c0 = Shared((1), 'm_c0')
        s_c0 = Shared((1), 's_c0')
        m_sigma = Shared((1), 'm_sigma')
        s_sigma = Shared((1), 's_sigma')

        # Define noise for model parameters
        z_w = srng.normal((D, K))
        z_r = srng.normal([K])
        z_gamma = srng.normal([1])
        z_gamma0 = srng.normal([1])
        z_c0 = srng.normal([1])
        z_sigma = srng.normal([1])

        # Define variational parameters
        # All model parameters have a log-normal variational posterior
        w = T.exp(m_w + z_w * s_w)
        r = T.exp(m_r + z_r * s_r)
        gamma = T.exp(m_gamma + z_gamma * s_gamma)
        gamma0 = T.exp(m_gamma0 + z_gamma0 * s_gamma0)
        c0 = T.exp(m_c0 + z_c0 * s_c0)
        sigma = T.exp(m_sigma + z_sigma * s_sigma)

        # Define random variables for mVNscan component
        z_y = srng.normal([D])
        z_k = srng.normal([K])
        z_eps = srng.normal()

        # For data given seqentially we need a different covariance matrix for each yn
        wwT = T.dot(w, w.T)
        cov = Shared((D, D), 'cov')
        cov = wwT + sigma[0] * T.identity_like(wwT)

        return mask, m, y, zero_y, zero2, zero, st, m_w, s_w, w, m_r, s_r, r, m_gamma, s_gamma, gamma, m_gamma0, \
               s_gamma0, gamma0, m_c0, s_c0, c0, m_sigma, s_sigma, sigma, z_y, z_k, z_eps, wwT, cov


    @staticmethod
    def _empirical(N, D, K, mnp, masknp):
        Shared = lambda shape, name: theano.shared(value=np.ones(shape, dtype=theano.config.floatX),
                                                   name=name, borrow=True)
        srng = T.shared_randomstreams.RandomStreams(seed=120)
        mask = Shared((D, N), 'mask')
        mask.set_value(masknp)
        m = T.as_tensor_variable(mnp)
        y = mask * m
        zero_y = T.as_tensor_variable(np.zeros((D, N)))
        zero2 = T.as_tensor_variable(np.zeros((D, D)))
        zero = T.as_tensor_variable(np.zeros(D))
        st = T.sum(T.neq(y, zero_y), axis=0)
        s = st.eval()

        scale = 1/((T.dot(mask,mask.T)) + T.ones((D,N)))
        cov = scale*(T.dot(y,y.T))
        eigval = T.abs_(T.min([T.min(T.nlinalg.eig(cov)[0]), 0]))
        #eigval = T.abs_(T.min(T.nlinalg.eig(cov)[0]))
        w = theano.tensor.slinalg.cholesky(cov + (eigval + 0.1)* T.eye(D))
        wwT = T.dot(w, w.T)

        # Define random variables for mVNscan component
        z_y = srng.normal([D])
        z_k = srng.normal([D])
        z_eps = srng.normal()
        return mask, m, y, zero_y, zero2, zero, st, scale, cov, w, eigval, wwT, z_y, z_k, z_eps

    @staticmethod
    def _lowrnk_emp(N, D, K, mnp, masknp):
        Shared = lambda shape, name: theano.shared(value=np.ones(shape, dtype=theano.config.floatX),
                                                   name=name, borrow=True)
        srng = T.shared_randomstreams.RandomStreams(seed=120)
        mask = Shared((D, N), 'mask')
        mask.set_value(masknp)
        m = T.as_tensor_variable(mnp)
        y = mask * m
        zero_y = T.as_tensor_variable(np.zeros((D, N)))
        zero2 = T.as_tensor_variable(np.zeros((D, D)))
        zero = T.as_tensor_variable(np.zeros(D))
        st = T.sum(T.neq(y, zero_y), axis=0)
        s = st.eval()

        scale = 1 / ((T.dot(mask, mask.T)) + T.ones((D, N)))
        emp_cov = scale * (T.dot(y, y.T))
        [U,S,V] = T.nlinalg.svd(emp_cov)
        rk = T.sum(S>0.2)
        cov = (U[:,0:rk].dot(T.nlinalg.diag(S[0:rk]))).dot(V[0:rk,:])
        eigval = T.abs_(T.min([T.min(T.nlinalg.eig(cov)[0]), 0]))
        cov = cov + (eigval + 0.1) * T.eye(D)
        print('so far so good')
        w = theano.tensor.slinalg.cholesky(cov)
        print('w calculated')
        wwT = T.dot(w, w.T)

        # Define random variables for mVNscan component
        z_y = srng.normal([D])
        z_k = srng.normal([D])
        z_eps = srng.normal()

        return mask, m, y, zero_y, zero2, zero, st, scale, cov, w, eigval, wwT, z_y, z_k, z_eps

    @staticmethod
    def _true(N,D,KTRUE,mnp,masknp,cov_data_name, w_data_name):
        Shared = lambda shape, name: theano.shared(value=np.ones(shape, dtype=theano.config.floatX),
                                                   name=name, borrow=True)
        srng = T.shared_randomstreams.RandomStreams(seed=120)
        mask = Shared((D, N), 'mask')
        mask.set_value(masknp)
        m = T.as_tensor_variable(mnp)
        y = mask * m
        zero_y = T.as_tensor_variable(np.zeros((D, N)))
        zero2 = T.as_tensor_variable(np.zeros((D, D)))
        zero = T.as_tensor_variable(np.zeros(D))
        st = T.sum(T.neq(y, zero_y), axis=0)
        s = st.eval()

        cov_true = np.loadtxt(cov_data_name, delimiter=',')
        cov = T.as_tensor_variable(cov_true)
        w_true = np.loadtxt(w_data_name, delimiter=',')
        w = T.as_tensor_variable(w_true.reshape((D,KTRUE)))
        wwT = T.dot(w, w.T)


        # Define random variables for mVNscan component
        z_y = srng.normal([D])
        z_k = srng.normal([KTRUE])
        z_eps = srng.normal()

        return mask, m, y, zero_y, zero2, zero, st, cov, w, wwT, z_y, z_k, z_eps


    @staticmethod
    def _sgld_updt(N, D, K, m_params, v_params, y, cov, mask, z_eps, gd_limit=500):
        """ functions to employ Stochastic Gradient Langevin Dynamics
            References
            ----------
            .. [1] Welling, M., & Teh, Y. W. (2011).
            Bayesian learning via stochastic gradient Langevin dynamics.
            (ICML-11) (pp. 681-688).

            Outputs:
                diffusion_fn (to sample for IDS)
                adadeltastep_fn (to move model parameters closer to MAP estimate)
            """
        [w, r, gamma, gamma0, c0, sigma] = m_params
        curr_log_joint = log_joint_fn(N, D, K, m_params, y, cov, mask)
        diffusion_updates = diffusion_helper_fn(curr_log_joint, m_params, z_eps)
        diffusion_fn = theano.function(inputs=[], updates=diffusion_updates)


        [results, m_param_updates] = theano.scan(fn=adadelta,
                                                 sequences=None,
                                                 non_sequences=[-curr_log_joint, w, r, gamma, gamma0, c0, sigma],
                                                 outputs_info=None,
                                                 n_steps=gd_limit)

        adadeltastep_fn = theano.function(inputs=[], updates=m_param_updates)

        return [diffusion_fn, adadeltastep_fn]

    @staticmethod
    def _svi_updt(N, D, K, m_params, v_params, y, cov, mask, z_eps, gd_limit=500):

        [m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma] = v_params
        [w, r, gamma, gamma0, c0, sigma] = m_params

        # Update function to get model parameters close to MAP
        curr_log_joint = log_joint_fn(N, D, K, m_params, y, cov, mask)


        [results, v_param_updates0] = theano.scan(fn=adadelta_svi,
                                                 sequences=None,
                                                 non_sequences=[-curr_log_joint,m_w, s_w, m_r, s_r, m_gamma, s_gamma,
                                                                m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma],
                                                 outputs_info=None,
                                                 n_steps=gd_limit)

        adadeltastep_fn = theano.function(inputs=[], updates=v_param_updates0)


        # Update function for SVI
        if any(not isinstance(p, theano.compile.SharedVariable) for p in v_params):
            raise ValueError("params must contain shared variables only.")

        elbo = elbo_fn(N, D, K, m_params, y, cov, mask, v_params)
        [results, v_param_updates] = theano.scan(fn=adadelta_svi,
                                                 sequences=None,
                                                 non_sequences=[-elbo, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0,
                                                                s_gamma0, m_c0, s_c0, m_sigma, s_sigma],
                                                 outputs_info=None,
                                                 n_steps=gd_limit)

        svi_adadeltastep_fn = theano.function(inputs=[], updates=v_param_updates)

        return [svi_adadeltastep_fn, adadeltastep_fn]



#Define functions for build_model_updates

def log_joint_scan_fn(n, llik, y, cov, mask):
    
    partial_cov = T.outer(mask[:, n], mask[:, n]) * cov + (1 - mask[:, n]) * T.identity_like(cov)
    llik += (-1 / 2.0) * T.log(T.nlinalg.Det()(partial_cov)) - (1 / 2.0) * T.dot(y[:, n].T, T.dot(
        T.nlinalg.MatrixInverse()(partial_cov), y[:, n]))

    return llik

def log_joint_fn(N, D, K,  m_params, y, cov, mask):

    w, r, gamma, gamma0, c0, sigma = m_params

    results, updates = theano.scan(fn=log_joint_scan_fn,
                                   sequences=np.arange(N),
                                   outputs_info=[dict(initial=np.float64(0), taps=[-1])],
                                   non_sequences=[y, cov, mask])

    log_joint = results[-1]

    log_joint += ((D * gamma * T.log(gamma))[0] * r).sum() - (D * T.gammaln(gamma[0] * r)).sum() + (
    (gamma[0] * r - 1) * T.log(w)).sum() - (gamma[0] * w).sum() + (
                gamma0 * T.log(c0) - K * T.gammaln(gamma0 / K) + (gamma0 / K - 1)[0] * (T.log(r)).sum() - (
                c0[0] * r).sum() - gamma - gamma0 - c0)[0]

    return log_joint

def entropy_fn(v_params):
    m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma = v_params
    ent_w = T.log(T.abs_(s_w)) + m_w
    ent_r = T.log(T.abs_(s_r)) + m_r
    ent_gamma = T.log(T.abs_(s_gamma)) + m_gamma
    ent_gamma0 = T.log(T.abs_(s_gamma0)) + m_gamma0
    ent_c0 = T.log(T.abs_(s_c0)) + m_c0
    ent_sigma  = T.log(T.abs_(s_sigma))+m_sigma
    entropy = ent_w.sum() + ent_r.sum() + ent_gamma + ent_gamma0 + ent_c0 + ent_sigma
    return entropy

def elbo_fn(N, D, K,  m_params, y, cov, mask, v_params):
    return log_joint_fn(N, D, K, m_params, y, cov, mask) + entropy_fn(v_params)[0]

def diffusion_helper_fn(curr_log_joint, m_params, z_eps, t=20):

    grads = theano.grad(curr_log_joint, m_params)
    updates=OrderedDict()

    for param, grad in zip(m_params, grads):

        step=(0.5*np.power((t+1), -1/3.0))*(grad/(T.sum(T.abs_(grad))))+ np.power((t+1), -1/3.0)*z_eps
        updates[param]=T.minimum(T.maximum((param+step).astype(theano.config.floatX), 0.00001*T.ones_like(param)), 1*T.ones_like(param))

    return updates


def adadelta(loss_or_grads, w, r, gamma, gamma0, c0, sigma, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """ 
    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """
    params = [w, r, gamma, gamma0, c0, sigma]

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        # accu = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        # updates[param] = param - learning_rate * update

        updates[param] = T.minimum(
            T.maximum((param - learning_rate * update).astype(theano.config.floatX), (1e-10) * T.ones_like(param)),
            10 * T.ones_like(param))
        # param = T.minimum(T.maximum((param - learning_rate * update).astype(theano.config.floatX), (1e-10)*T.ones_like(param)), 10*T.ones_like(param))
        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new
        # delta_accu = delta_accu_new

    return updates

def adadelta_svi(loss_or_grads, m_w, m_r, m_gamma, m_gamma0, m_c0, m_sigma, s_w, s_r, s_gamma, s_gamma0, s_c0, s_sigma,
             learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """
    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """
    params = [m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma]

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        # accu = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        # updates[param] = param - learning_rate * update

        updates[param] = T.minimum(
            T.maximum((param - learning_rate * update).astype(theano.config.floatX), -10*T.ones_like(param)),
            10 * T.ones_like(param))
        # param = T.minimum(T.maximum((param - learning_rate * update).astype(theano.config.floatX), (1e-10)*T.ones_like(param)), 10*T.ones_like(param))
        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new
        # delta_accu = delta_accu_new

    return updates





def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)