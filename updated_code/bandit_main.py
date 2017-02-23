from __future__ import division
import bandit_utils, bandit_theano
from bandit_utils import Bandit
from bandit_theano import Model
import numpy as np
import theano
import theano.tensor as T
import time
from collections import OrderedDict
import sys


sys.setrecursionlimit(50000)


def recommender(random_seed, save_name,read_in_data, data_name, cov_data_name, w_data_name, KTRUE=1, K=20, N=100, D=50, maxit_factor=2,
                bandit_algorithm='ts', posterior_method = 'sgld', n_blocks=10, gd_limit=250, verbose=0, sample_heavy = 0):
    maxit = int(np.floor(maxit_factor*(KTRUE*(N+D-KTRUE))))
    n_iter = int(np.floor(maxit / n_blocks))
    if read_in_data:
        mnp = np.loadtxt(data_name, delimiter=',')
    else:
        np.random.seed(seed=random_seed)
        wnp = np.random.uniform(low=0.0, high=1.0, size=(D, KTRUE))
        # Introduce some complexity into w
        # w     = np.random.beta(a = 2, b = 5, size = (D,KTRUE))
        # maskw = stats.bernoulli.rvs(0.8, size=(D,KTRUE))
        # w     = np.multiply(w,maskw)
        var = 0.1
        covnp = wnp.dot(wnp.T) + var * np.eye(D)
        mnp = np.random.multivariate_normal(np.zeros(D), covnp, N).T

    random_w_init = np.random.uniform(low=0.01, high=2.0, size=(D, K))
    random_r_init = np.random.uniform(low=0.01, high=2.0, size=K)

    # Construct mask matrix, initial observations/ratings

    prob = (maxit / (N * D)) / (3*n_blocks)
    observe = np.random.permutation(np.arange(N * D))[1:int(np.floor(prob*N*D))]
    masknp = np.zeros(N * D)
    masknp[observe] = 1
    if sample_heavy:
        observerow = np.random.randint(0, high=D, size=N)
        observecol = np.arange(N)
        masknp = masknp.reshape((D, N))
        masknp[observerow, observecol[0:np.size(observerow)]] = 1
    masknp = masknp.reshape((D, N))
    initial_obs = np.multiply([masknp != 0], mnp).flatten()
    ratings = initial_obs[initial_obs != 0].tolist()
    print('number of initial obs is: {}'.format(np.size(ratings)))

    # Build Model and Model Updates for SGLD
    if posterior_method == 'sgld':
        [mask, m, y, zero_y, zero2, zero, st, w, r, gamma, gamma0, c0, sigma, z_y, z_k, z_eps,
         wwT, cov] = Model.build(N, D, K, mnp, masknp, random_w_init, random_r_init, cov_data_name, w_data_name, posterior_method)
        m_params = [w, r, gamma, gamma0, c0, sigma]
        v_params = theano.shared(np.zeros(1))
        [diffusion_fn, adadeltastep_fn] = Model.build_updates(N, D, K, m_params, v_params, y, cov, mask, z_eps,
                                                              posterior_method, gd_limit)
        m_w = theano.shared(np.zeros(1))
        s_w = theano.shared(np.zeros(1))
        m_r = theano.shared(np.zeros(1))
        s_r =theano.shared(np.zeros(1))
        m_gamma=theano.shared(np.zeros(1))
        s_gamma=theano.shared(np.zeros(1))
        m_gamma0 = theano.shared(np.zeros(1))
        s_gamma0=theano.shared(np.zeros(1))
        m_c0 = theano.shared(np.zeros(1))
        s_c0= theano.shared(np.zeros(1))
        m_sigma = theano.shared(np.zeros(1))
        s_sigma = theano.shared(np.zeros(1))
        svi_adadeltastep_fn = theano.shared(np.zeros(1))
        eig_or_rank = 0
    elif posterior_method == 'svi':
        [mask, m, y, zero_y, zero2, zero, st, m_w, s_w, w, m_r, s_r, r, m_gamma, s_gamma, gamma, m_gamma0, s_gamma0, \
        gamma0, m_c0, s_c0, c0, m_sigma, s_sigma, sigma, z_y, z_k, z_eps, wwT, cov] = Model.build(N, D, K, mnp, masknp,
                                                                                                  random_w_init,
                                                                                                  random_r_init,
                                                                                                  cov_data_name, w_data_name,
                                                                                                  posterior_method)
        m_params = [w, r, gamma, gamma0, c0, sigma]
        v_params = [m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma]

        [svi_adadeltastep_fn, adadeltastep_fn] = Model.build_updates(N, D, K, m_params, v_params, y, cov, mask, z_eps,
                                                              posterior_method, gd_limit)
        diffusion_fn = 0
        eig_or_rank = 0

    elif posterior_method == 'empirical' or posterior_method == 'low_rank_emp':
        [mask, m, y, zero_y, zero2, zero, st, scale, cov, w, eig_or_rank, wwT,z_y, z_k, z_eps ] = Model.build(N, D, K, mnp, masknp, random_w_init,
                                                                         random_r_init, cov_data_name, w_data_name, posterior_method)
        m_w = theano.shared(np.zeros(1))
        s_w = theano.shared(np.zeros(1))
        m_r = theano.shared(np.zeros(1))
        s_r = theano.shared(np.zeros(1))
        m_gamma = theano.shared(np.zeros(1))
        s_gamma = theano.shared(np.zeros(1))
        m_gamma0 = theano.shared(np.zeros(1))
        s_gamma0 = theano.shared(np.zeros(1))
        m_c0 = theano.shared(np.zeros(1))
        s_c0 = theano.shared(np.zeros(1))
        m_sigma = theano.shared(np.zeros(1))
        s_sigma = theano.shared(np.zeros(1))
        svi_adadeltastep_fn = theano.shared(np.zeros(1))
        adadeltastep_fn = theano.shared(np.zeros(1))
        diffusion_fn = 0
        r = 0
        gamma =0
        gamma0 = 0
        c0 = 0
        sigma = theano.shared(np.zeros(1))
        sigma.set_value([0.1])
        print('build model')
    elif posterior_method == 'true':
        [mask, m, y, zero_y, zero2, zero, st,  cov, w, wwT, z_y, z_k, z_eps] = Model.build(N, D, KTRUE, mnp, masknp, random_w_init,
                                                                         random_r_init, cov_data_name, w_data_name, posterior_method)

        m_w = theano.shared(np.zeros(1))
        s_w = theano.shared(np.zeros(1))
        m_r = theano.shared(np.zeros(1))
        s_r = theano.shared(np.zeros(1))
        m_gamma = theano.shared(np.zeros(1))
        s_gamma = theano.shared(np.zeros(1))
        m_gamma0 = theano.shared(np.zeros(1))
        s_gamma0 = theano.shared(np.zeros(1))
        m_c0 = theano.shared(np.zeros(1))
        s_c0 = theano.shared(np.zeros(1))
        m_sigma = theano.shared(np.zeros(1))
        s_sigma = theano.shared(np.zeros(1))
        svi_adadeltastep_fn = theano.shared(np.zeros(1))
        adadeltastep_fn = theano.shared(np.zeros(1))
        diffusion_fn = 0
        r = 0
        gamma = 0
        gamma0 = 0
        c0 = 0
        sigma = theano.shared(np.zeros(1))
        sigma.set_value([0.1])
        eig_or_rank = 0
    else:
        raise Exception(posterior_method + 'posterior estimation method not implemented.')


    action_ordering = np.zeros(maxit)
    count = 0
    n_try = 0
    threshold = 1.2 * maxit

    t_initial = time.time()
    while count <= maxit and n_try <= threshold:
        [count, n_try, masknp, action_ordering, masknp, st, mask, m, y, zero_y, zero2, zero,m_w, s_w, m_r, s_r, m_gamma,
         s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma, w, r, gamma, gamma0, c0,
        sigma, z_y, z_k, z_eps, wwT, cov, eig_or_rank] = bandit_per_loop(mnp, count, n_try, n_iter, N, D, masknp, action_ordering, verbose, st, mask, m, y, zero_y, zero2,
                    zero, w, r, gamma, gamma0, c0, sigma, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0,
                    m_c0, s_c0, m_sigma, s_sigma, z_y, z_k, z_eps, wwT, cov, eig_or_rank, diffusion_fn, adadeltastep_fn,
                    svi_adadeltastep_fn, bandit_algorithm, posterior_method)
        """
        covnpcurr = cov.eval()
        scale = covnp[1,1]/covnpcurr[1,1]
        covnpcurr *= scale
        wnp = w.eval()
        max_val = np.max(wnp.flatten())
        print('max(w) = {}'.format(max_val))
        print('Relative covariance estimate error is: {}.'.format(abs(covnp-covnpcurr).sum()/(abs(covnp).sum())))
        """
        print("Bandit is: {} percent complete in {} seconds.".format((100 * count) / maxit, time.time() - t_initial))
        print("Action set has chosen {} entries.".format((action_ordering!=0).sum()))

    print("Bandit is complete.")
    mnpf = mnp.flatten()
    for ii in range(maxit):
        ratings.append(mnpf[int(action_ordering[ii])])


    reward = np.cumsum(ratings)

    best = mnp.flatten()
    best.sort()
    best[:] = best[::-1]
    best = np.cumsum(best)

    random_reward = np.zeros(np.size(mnp.flatten()))
    for i in range(10):
        random = mnp.flatten()
        random = np.random.permutation(random)
        random_reward += np.cumsum(random) / 10

    np.savetxt(save_name, reward, delimiter=',')


    return action_ordering, reward,random_reward, best


def ind2sub(array_shape, ind):
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = np.floor(ind / array_shape[1])
    cols = ind % array_shape[1]
    return (int(rows), int(cols))


def bandit_per_loop(mnp, count, n_try, n_iter, N, D, masknp, action_ordering, verbose, st, mask, m, y, zero_y, zero2,
                    zero, w, r, gamma, gamma0, c0, sigma, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0,
                    m_c0, s_c0, m_sigma, s_sigma, z_y, z_k, z_eps, wwT, cov, eig_or_rank, diffusion_fn, adadeltastep_fn,
                    svi_adadeltastep_fn, bandit_algorithm, posterior_method):
    s = st.eval()
    tt = time.time()

    if posterior_method == 'sgld' or posterior_method=='svi':
        adadeltastep_fn()

    if posterior_method == 'svi':
        svi_adadeltastep_fn()


    if verbose:
        print("adadeltastep_fn time is: {} seconds.".format(time.time() - tt))

    tt = time.time()
    print('about to determine actions')
    actions = Bandit.pull(mnp, y, n_iter, mask, cov, s, w, z_y, z_k, st, N, zero, zero2, D, sigma, diffusion_fn,
                          bandit_algorithm, posterior_method)
    if verbose:
        print("action set determined in: {} seconds.".format(time.time() - tt))

    for nn in np.arange(n_iter):
        action = int(actions[nn])
        n_try += 1
        if action in action_ordering:
            annie = 1
        else:
            mf = masknp.flatten()
            mf[action] = 1
            masknp = mf.reshape((D, N))
            mask.set_value(masknp)
            if count<np.size(action_ordering):
                action_ordering[count] = action
                count += 1

    return [count, n_try, masknp, action_ordering, masknp, st, mask, m, y, zero_y, zero2, zero,m_w, s_w, m_r, s_r,
            m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, m_sigma, s_sigma, w, r, gamma, gamma0, c0, sigma, z_y,
            z_k, z_eps, wwT, cov, eig_or_rank]
