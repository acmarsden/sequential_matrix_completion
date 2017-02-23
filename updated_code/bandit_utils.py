from __future__ import division
import numpy as np
import theano.tensor as T
import theano

class Bandit(object):
    @staticmethod
    def pull(mnp, y, n_iter, mask, cov, s, w, z_y, z_k, st, N, zero, zero2, D, sigma, diffusion_fn,
                          bandit_algorithm, posterior_method):
        if bandit_algorithm == 'ids':
            return Bandit._ids(mnp, y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma, diffusion_fn, N,
                                              n_iter, posterior_method)
        elif bandit_algorithm == 'ts':
            return Bandit._ts(mnp, y, mask, cov,s, w, z_y, z_k, st, N, n_iter, zero, zero2, D, sigma)
        else:
            raise Exception(bandit_algorithm + 'bandit algorithm not implemented.')


    @staticmethod
    def _ids(mnp, y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma, diffusion_fn, N,
                                              n_iter, posterior_method):

        regret_info_ratio = compute_info_gain(y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma, diffusion_fn, N,
                                              posterior_method, n_iter)

        return regret_info_ratio


    @staticmethod
    def _ts(mnp, y, mask, cov, s, w, z_y, z_k, st, N, n_iter, zero, zero2, D, sigma):
        print('nnz')
        print((mask.eval()).sum())
        y_est, updates = theano.scan(fn=mvn_scan,
                                      sequences=T.arange(N),
                                      outputs_info=None,
                                      non_sequences=[y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma])

        [value, index] = T.max_and_argmax(y_est.T)

        action_list = np.zeros(n_iter)
        for nn in range(n_iter):
            action = index.eval()
            action_list[nn] = int(action)
            #print('true value')
            #mnpf = mnp.flatten()
            #print(mnpf[int(action)])

        return action_list


def mvn_scan(n,y, mask, cov, w, z_y, z_k, s, zero, zero2, D, sigma):
    R = 10

    #Problematic Method
    """
    # construct unobs a vector of 1s and 0s where the ith coord is a 1 if we haven't seen the ith coord of y_n    
    unobs = T.eq(y[:, n], zero)
    # construct covariance of the observed entries where the rows/columns with nothing have a 1 on diag (so invertible)
    idxs          = T.neq(y[:, n], zero).nonzero()
    y_obs         = y[:, n][idxs]
    idxs_mat      = T.neq(T.outer(y[:, n], y[:, n]), zero2).nonzero()
    temp_cov      = cov[idxs_mat].reshape((s[n], s[n]))
    temp_cov_inv  = T.nlinalg.MatrixInverse()(temp_cov)
    cov_unobs_obs = (T.outer(unobs, T.neq(y[:, n], zero))) * cov
    cov_unobs_obs = cov_unobs_obs[:, idxs].reshape((D, s[n]))
    """
    y_obs = y[:,n]
    unobs         = 1-mask[:,n]
    temp_cov      = T.outer(mask[:,n], mask[:,n])*cov + (1-mask[:,n])*T.identity_like(cov)
    temp_cov_inv  = T.nlinalg.MatrixInverse()(temp_cov)
    cov_unobs_obs = T.outer(unobs, mask[:,n])*cov

    # draw the mean vector dummy from N(0, wwT+sigma^2I) using computationally fast trick
    dummy_results, dummy_updates = theano.scan(lambda prior_result, sigma, z_y, w, z_k:
                                               T.sqrt(sigma)[0] * z_y + T.dot(w, z_k) + prior_result,
                                               sequences= None,
                                               outputs_info=T.zeros(D),
                                               non_sequences=[sigma, z_y, w, z_k],
                                               n_steps=R)

    dummy = dummy_results[-1]
    dummy /= R
    #dummy_obs = dummy[idxs]
    dummy_obs = dummy*mask[:,n]
    dummy_unobs = unobs * dummy
    y_est = dummy_unobs + T.dot(T.dot(cov_unobs_obs, temp_cov_inv), (y_obs - dummy_obs))
    y_est = (y_est * unobs) - (1e6) * mask[:, n]
    
    # Uncomment to estimate algorithm ability in estimating true matrix
    # y_est        = y_est*unobs + y[:,n]*mask[:,n]

    return y_est


def compute_info_gain(y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma, diffusion_fn, N, posterior_method, n_iter=100, n_theta=20,
                      n_pdf_theta=100, q_range=10, max_value=20, min_value=-20, n_bins=100, R=10):

    theta_array = np.zeros((n_theta, D, D))
    optind_array = np.zeros((N * D))
    theta_opt_key = np.zeros(n_theta)
    pdf_theta = np.zeros(n_theta)
    q_theta_temp = np.zeros((n_theta, D, N, n_bins))
    q_theta = np.zeros((n_theta, n_bins, D, N))

    #y_avg = np.zeros((D, N))
    
    avg_count = 0
    
    for ii in range(n_theta):

        if posterior_method == 'sgld':
            diffusion_fn()

        curr_cov = cov.eval()
        theta_array[ii, :, :] = curr_cov
        fixed_cov = T.as_tensor_variable(curr_cov)
        y_est, updates = theano.scan(fn=mvn_scan,
                                     sequences=T.arange(N),
                                     outputs_info=None,
                                     non_sequences=[y, mask, cov, w, z_y, z_k, st, zero, zero2, D, sigma])

        mu = np.zeros((D,N))
        for jj in range(q_range):

            y_temp = y_est.eval().T
            y_temp[y_temp == -1e6] = 0
            #y_avg += y_temp
            mu += y_temp
            avg_count += 1
            # print('post')
            if np.max(y_temp.flatten()) > max_value:
                print('In bandit_utils.compute_info_gain: bin range too small')
                print(np.max(y_temp))
            if np.min(y_temp.flatten()) < min_value:
                print('In bandit_utils.compute_info_gain: bin range too small')
                print(np.min(y_temp))
            bin_record_temp = np.floor(((y_temp - min_value) / (max_value - min_value)) * n_bins)
            q_theta_temp[ii, :, :, jj] = bin_record_temp

        for kk in range(n_bins):
            for jj in range(q_range):
                q_theta[ii, kk, :, :] += [q_theta_temp[ii, :, :, jj] == kk][0]
        q_theta[ii, :, :, :] /= q_range

    for ii in range(n_theta):
        for nn in range(N):
            for dd in range(D):
                q_theta[ii, :, dd, nn] /= np.abs(q_theta[ii, :, dd, nn]).sum()

    for jj in range(n_pdf_theta):
        if posterior_method == 'sgld':
            diffusion_fn()

        curr_cov2 = cov.eval()
        diff = 30
        point = -10
        for ii in range(n_theta):
            diff_new = np.sum(np.abs(curr_cov2 - theta_array[ii, :, :])) / (N * D)
            if diff_new < diff:
                diff = diff_new
                point = ii
        if point != -10:
            # print(point)
            pdf_theta[point] += 1
    pdf_theta += 1
    pdf_theta = pdf_theta / pdf_theta.sum()
    # print('pdf_theta')
    # print(pdf_theta)



    # Compute pdf_opt
    for ii in range(n_theta):
        y_temp = np.zeros((D, N))
        for yy in range(n_bins):
            y_temp += q_theta[ii, yy, :, :] * ((yy / n_bins) * (max_value - min_value) + min_value)
        ind = np.argmax(y_temp.flatten())
        optind_array[ind] += 1
        theta_opt_key[ii] = ind

    list_opt = np.multiply([optind_array != 0], np.arange(N * D))
    list_opt = list_opt[0]
    list_opt = list_opt[list_opt != 0]
    n_opt = np.size(list_opt)
    pdf_opt = np.zeros(n_opt)
    for ii in range(n_opt):
        opt_ind = list_opt[ii]
        theta_corr = [theta_opt_key == opt_ind]
        theta_corr = theta_corr[0]
        for jj in range(n_theta):
            if theta_corr[jj]:
                pdf_opt[ii] += pdf_theta[jj]

    pdf_opt = pdf_opt / (np.sum(pdf_opt))

    pdf_a_y = np.zeros((n_bins, D, N))

    for yy in range(n_bins):
        for ii in range(n_theta):
            pdf_a_y[yy, :, :] += pdf_theta[ii] * q_theta[ii, yy, :, :]

    pdf_opt_y = np.zeros((n_opt, n_bins, D, N))

    for yy in range(n_bins):
        for ii in range(n_opt):
            opt_ind = list_opt[ii]
            theta_corr = [theta_opt_key == opt_ind];
            theta_corr = theta_corr[0]
            for jj in range(n_theta):
                if theta_corr[jj]:
                    pdf_opt_y[ii, yy, :, :] += q_theta[jj, yy, :, :]
            pdf_opt_y[ii, yy, :, :] *= pdf_opt[ii]

    # Renormalize- for some reason it sums up to not quite 1
    for nn in range(N):
        for dd in range(D):
            pdf_opt_y[:, :, dd, nn] /= np.sum(pdf_opt_y[:, :, dd, nn])

    R_star = 0
    for ii in range(n_opt):
        opt_ind = list_opt[ii]
        theta_corr = [theta_opt_key == opt_ind];
        theta_corr = theta_corr[0]
        for jj in range(n_theta):
            if theta_corr[jj]:
                # print(jj)
                # print('yes1')
                [row, col] = ind2sub((D,N), opt_ind)
                for yy in range(n_bins):
                    R_star += pdf_theta[jj] * q_theta[jj, yy, row, col] * (
                    (yy / n_bins) * (max_value - min_value) + min_value)

    entropy_change = np.zeros((D, N))
    # MAKE FASTER LATER

    for nn in range(N):
        for dd in range(D):
            for yy in range(n_bins):
                if pdf_a_y[yy, dd, nn] != 0:
                    for ii in range(n_opt):
                        if pdf_opt_y[ii, yy, dd, nn] != 0:
                            entropy_change[dd, nn] += pdf_opt_y[ii, yy, dd, nn] * np.log(
                                pdf_opt_y[ii, yy, dd, nn] / (pdf_opt[ii] * pdf_a_y[yy, dd, nn]))

    regret_mat = np.zeros((D, N))

    for ii in range(n_theta):
        for yy in range(n_bins):
            regret_mat += pdf_theta[ii] * q_theta[ii, yy, :, :] * ((yy / n_bins) * (max_value - min_value) + min_value)

    #y_avg /= avg_count

    regret_mat = R_star - regret_mat
    metric = np.divide(np.power(regret_mat, 2), entropy_change)
    metric_flat = metric.flatten()
    action_list = np.zeros(n_iter)
    set_max = np.max(metric_flat)
    for nn in range(n_iter):
        action = np.argmin(metric_flat)
        metric_flat[action] = set_max + 1
        action_list[nn] = int(action)

    return action_list


def ind2sub(array_shape, ind):
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = np.floor(ind / array_shape[1])
    cols = ind % array_shape[1]
    return (int(rows), int(cols))