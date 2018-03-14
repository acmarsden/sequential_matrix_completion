import bandit_main
import numpy as np

ranks = [1,5,10,20]
jj = 0
ii = 0
for jj in range(3):
    jj+=1
    kk = ranks[jj]

    if kk == 1:
        sh = 0
    else:
        sh = 1
    data_name = 'SYNTHETIC_MATRICES/M'+str(ii)+'K' + str(kk) + '.csv'
    print('reading matrix {}.'.format(data_name))


    print('reading matrix {}.'.format(data_name))
    save_name = 'SVI_IDS/2svi_ids_k'+str(kk)+'_'+str(ii) + '.csv'
    [actions, rewards, random_reward, best_rewards, entropy] = bandit_main.recommender(10, save_name,1, data_name, 0, 0, KTRUE = kk,K=20,N=100, D=50, maxit_factor=2, bandit_algorithm='ids',posterior_method = 'svi', n_blocks=40, gd_limit=500, verbose=0, sample_heavy=sh)
    np.savetxt('ids_ent'+str(kk)+str(ii)+'.csv', entropy, delimiter=',')
    np.savetxt('ids_actions'+str(kk)+str(ii)+'.csv', actions, delimiter=',')


    save_name = 'SVI_TS/2svi_ts_k'+str(kk)+'_'+str(ii) + '.csv'
    [actions, rewards, random_reward, best_rewards, entropy] = bandit_main.recommender(10, save_name,1, data_name, 0, 0, KTRUE = kk,K=20,N=100, D=50, maxit_factor=2, bandit_algorithm='ts',posterior_method = 'svi', n_blocks=40, gd_limit=500, verbose=0, sample_heavy=sh)
    np.savetxt('ts_ent'+str(kk)+str(ii)+'.csv', entropy, delimiter=',')
    np.savetxt('ts_actions'+str(kk)+str(ii)+'.csv', actions, delimiter=',')
