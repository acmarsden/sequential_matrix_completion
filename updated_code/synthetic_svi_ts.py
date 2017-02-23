import bandit_main
import numpy as np

ranks = [1,5,10,20]

for jj in range(4):
    kk = ranks[jj]
    if kk == 1:
        sh = 0
    else:
        sh = 1
    for ii in range(5):
        data_name = 'SYNTHETIC_MATRICES3/M'+str(ii)+'K' + str(kk) + '.csv'
        print('reading matrix {}.'.format(data_name))
        save_name = 'SVI_TS3/svi_ts_k'+str(kk)+'_'+str(ii) + '.csv'
	if kk ==1:
		[actions, rewards, random_reward, best_rewards] = bandit_main.recommender(10, save_name,1, data_name, 0, 0, KTRUE = kk,K=20,N=100, D=50, maxit_factor=2, bandit_algorithm='ts',posterior_method = 'svi', n_blocks=40, gd_limit=200, verbose=0, sample_heavy=sh)
