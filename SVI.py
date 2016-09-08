import os.path
import numpy as np
import theano
import theano.tensor as T
import lasagne 
import time
import theano.sandbox.cuda.basic_ops as C

TOTAL_TIME_1=time.clock()

#Define Dimensions and initial parameters
N = 50
D = 50

#Threshold Rank
THRESHOLD_RANK = 20

#True Rank of the partial_covariance_matrixariance Matrix
TRUE_RANK=10

#COVARIANCE_MATRIX=wwT+VAR*identity
VAR=0.1

#Variance in the noise of the observation
ERROR_VAR=1

#BETA discount factor
BETA=0

#Set how many iterations to perform variational inference and how many entries to observe initially here. 
NUM_ITERATIONS=20
NUM_OBSERVATIONS_TOTAL=THRESHOLD_RANK*(N+D-THRESHOLD_RANK)
NUM_INITIAL_RANDOM_OBSERVATIONS=int(np.floor((NUM_OBSERVATIONS_TOTAL/NUM_ITERATIONS)))

#To average out (maybe remove)
R=100


#RANDOM and TRUTH are for testing with forcing values of covariance_matrix (wwT+sigmaI) to be RANDOM or the 
#true covariance matrix
RANDOM=0
TRUTH=0


#Lists for data analysis. elbo_list slows down the algorithm to compute
CONSTRUCT_ELBO_LIST=0
index_list=[]
ratings=[]
row_list=[]
elbo_list=[]
frobenius_error_list=[]
rating_error=[]

#Define a stream of random numbers
seed_number=np.random.randint(low=100, high=1000)
srng = T.shared_randomstreams.RandomStreams(seed=seed_number)
ERROR=ERROR_VAR*srng.normal([1])

#This is simulated data. For the bandit we pass a matrix with some observed entries and some not.
#load_path= '/local/data/public/am2648'
#name_of_file= 'MATRIX_K'+str(TRUE_RANK)
#completeName=os.path.join(load_path, name_of_file+'.csv')
#with file(completeName, 'r') as infile:
#    true_full_matrix= np.loadtxt(infile, delimiter=',')
#true_full_matrix=np.loadtxt("MATRIX_K"+str(TRUE_RANK)+'.csv', delimiter=',')

true_full_matrix=np.random.randint(0, 1, size=((D,N)))
w_true=np.random.uniform(0,1,(D,TRUE_RANK))
mean=np.zeros(D)
true_covariance_matrix=np.dot(w_true, np.transpose(w_true))+VAR*np.identity(D)
true_full_matrix=np.random.multivariate_normal(mean, true_covariance_matrix ,N)
true_full_matrix=np.transpose(true_full_matrix)
observation_mask1=np.empty((D,N))
observation_mask1[:]=np.NAN

observed_matrix=np.empty((D,N))
observed_matrix[:]=np.NAN
observed_count=0
y_comparison=[]
matrix_sample_count=np.zeros((D,N))
matrix_sample_count_best=np.zeros((D,N))
matrix_sample_count_random=np.zeros((D,N))
col_list=[]

if BETA==0:
    observed_entries=[]
    observed_entry_index=[]
    while observed_count<NUM_INITIAL_RANDOM_OBSERVATIONS:
        d= np.random.randint(0,high=D)
        n=np.random.randint(0, high=N)
        observation_mask1[d,n]=1
        matrix_sample_count[d,n]+=1
        matrix_sample_count_best[d,n]+=1
        matrix_sample_count_random[d,n]+=1
        observed_count+=1
        observed_entries.append(true_full_matrix[d,n])
        observed_entry_index.append((d,n))
        index_list.append(n*d+n)
        row_list.append(d)
        col_list.append(n)
        ratings.append(true_full_matrix[d,n]*(BETA**(matrix_sample_count[d,n])))

else:
    observed_count=0
    while observed_count<NUM_INITIAL_RANDOM_OBSERVATIONS:
        d= np.random.randint(0,high=D)
        n=np.random.randint(0, high=int(N/2))
        observation_mask1[d,n]=1
        matrix_sample_count[d,n]+=1
        matrix_sample_count_best[d,n]+=1
        matrix_sample_count_random[d,n]+=1
        observed_count+=1

#observed_matrix=true_full_matrix*observation_mask1

            

#Define Theano Variables
Shared = lambda shape,name: theano.shared(value = np.ones(shape,dtype=theano.config.floatX),
                                          name=name,borrow=True) 

observation_mask=Shared((D,D), 'observation_mask')
incomplete_identity_matrix=Shared((D,D), "incomplete_identity_matrix")
theano_observed_matrix=Shared((D,N), "theano_observed_matrix")
is_observed_matrix=Shared((D,N), "is_observed_matrix")
is_observed_matrix=Shared((D,N), "is_observed_matrix")

#Define variational parameters
m_w=Shared((D,THRESHOLD_RANK), "m_w")
s_w=Shared((D,THRESHOLD_RANK), "s_w")
m_r=Shared((THRESHOLD_RANK), "m_r")
s_r=Shared((THRESHOLD_RANK), "s_r")
m_gamma=Shared((1), "m_gamma")
s_gamma=Shared((1), "s_gamma")
m_gamma0=Shared((1), "m_gamma0")
s_gamma0=Shared((1), "s_gamma0")
m_c0=Shared((1), "m_c0")
s_c0=Shared((1), "s_c0")
msigma=Shared((1), "msigma")
ssigma=Shared((1), "ssigma")

#Define model parameters and other RANDOM variables (zY, zK, error, covariance_matrix_RANDOM)
zw= srng.normal((D,THRESHOLD_RANK))
zr= srng.normal([THRESHOLD_RANK])
zgamma= srng.normal([1])
zgamma0= srng.normal([1])
zc0= srng.normal([1])
zsigma=srng.normal([1])
zY=srng.normal([D,N])
zK=srng.normal([THRESHOLD_RANK,N])
covariance_matrix_random=srng.uniform(size=(D,D), low=0, high=1000)

#All variables have a log-normal variational posterior
w=T.exp(m_w+0.1*s_w)
r=T.exp(m_r + zr*s_r)
gamma=T.exp(m_gamma + zgamma*s_gamma)
gamma0=T.exp(m_gamma0 + zgamma0*s_gamma0)
c0=T.exp(m_c0 + zc0*s_c0)
sigma=0.1
#sigma=T.exp(msigma +zsigma*ssigma)


#For data given seqentially we need a different covariance matrix for each yn
partial_covariance_matrix=Shared((D,D), "partial_covariance_matrix")
partial_covariance_matrix.set_value(np.eye(D).astype(np.float64))
wwT=Shared((D,D), "wwT")
wwT=T.dot(w, w.T)
covariance_matrix=Shared((D,D), "covariance_matrix")
covariance_matrix=wwT+sigma*T.identity_like(wwT)
partial_covariance_matrix=observation_mask*covariance_matrix+incomplete_identity_matrix


#Define lists
v_params= [m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma]
m_params = [w, r, gamma, gamma0, c0, sigma]

#Define Functions for Variational Inference
def Sub2Ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    #ind[ind < 0] = -1
    #ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def Ind2Sub(array_shape, ind):
    #ind[ind < 0] = -1
    #ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind / array_shape[1])
    cols = ind % array_shape[1]
    return (int(rows), int(cols))


def Entropy(v_params):
    m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma= v_params
    ent_w=(T.log(T.abs_(s_w))+m_w)
    ent_r=(T.log(T.abs_(s_r))+m_r)
    ent_gamma=(T.log(T.abs_(s_gamma))+m_gamma)
    ent_gamma0=(T.log(T.abs_(s_gamma0))+m_gamma0)
    ent_c0=(T.log(T.abs_(s_c0))+m_c0)
    ent_sigma=(T.log(T.abs_(ssigma))+msigma)
    entropy=ent_w.sum()+ent_r.sum()+ent_gamma+ent_gamma0+ent_c0+ent_sigma
    return(entropy)

def LogJointScanFn(n, log_likelihood, is_observed_matrix, theano_observed_matrix):
    #For each y_n, create "observation_mask" and "Incomplete Id" such that observation_mask*(wwT+*Id)+incomplete_identity_matrix gives
    #the identity matrix with the covariance of the observed entries of y_n included, this has the same determinant
    #as the marginal covariance matrix of the observed entries
    numpy_incomplete_identity_matrix=np.zeros((D,D))
    observation_mask=(T.outer(is_observed_matrix[:,n], is_observed_matrix[:,n]).reshape((D,D)))
    incomplete_identity_matrix=(T.floor(T.sum(1-observation_mask, axis=0)/D)*T.identity_like(observation_mask))
    partial_covariance_matrix=(observation_mask*covariance_matrix+incomplete_identity_matrix)
    log_likelihood+=((-1/2.0)*T.log(T.nlinalg.Det()(partial_covariance_matrix))-(1/2.0)*T.dot(theano_observed_matrix[:,n].T, T.dot(T.nlinalg.MatrixInverse()(partial_covariance_matrix), theano_observed_matrix[:,n])))
    return log_likelihood

def LogJ(m_params,v_params,theano_observed_matrix):
    m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma=v_params
    w, r, gamma, gamma0, c0, sigma= m_params
    is_observed_matrix_numpy=~np.isnan(observed_matrix)
    is_observed_matrix.set_value(is_observed_matrix_numpy.astype(np.float64))
    theano_observed_matrix.set_value((np.nan_to_num(is_observed_matrix_numpy*observed_matrix)).astype(np.float64))
    log_j_t0=time.clock()
    results, updates = theano.scan(fn=LogJointScanFn,
                                   sequences = [dict(input=np.arange(N) ,taps=[-1])],
                                   outputs_info=[dict(initial= np.float64(0) ,taps=[-1])],
                                   non_sequences=[is_observed_matrix, theano_observed_matrix])
    log_joint=results[-1]
    log_joint2= (((D*gamma*T.log(gamma))[0]*r).sum()-(D*T.gammaln(gamma[0]*r)).sum()+((gamma[0]*r-1)*T.log(w)).sum()-(gamma[0]*w).sum() + (gamma0*T.log(c0)-THRESHOLD_RANK*T.gammaln(gamma0/THRESHOLD_RANK)+(gamma0/THRESHOLD_RANK-1)[0]*(T.log(r)).sum()-(c0[0]*r).sum()-gamma-gamma0-c0)[0])
    log_joint += log_joint2
 
    return(log_joint)
    
def ELBO(m_params,v_params, theano_observed_matrix):
    return(LogJ(m_params,v_params, theano_observed_matrix)+Entropy(v_params)[0])

#initialize AdaDeltaStep so that it isn't initialized each time the function is called.
if RANDOM+TRUTH==0:
    elbo=ELBO(m_params,v_params, theano_observed_matrix)
    entropy=Entropy(v_params)
    log_joint=LogJ(m_params,v_params, theano_observed_matrix)
    paramt0=time.clock()
    vParamUpdates=lasagne.updates.adadelta(-elbo,v_params)
    paramt1=time.clock()
    AdaDeltaStep=theano.function(inputs=[], updates=vParamUpdates)

#Main Variational Inference Function
def GPFA(observed_matrix, limit, CONSTRUCT_ELBO_LIST, elbo_list):
    #Repeat Code as above to construct the partial covariance matrices, also construct is_observed_matrix
    #which is a matrix with a 1 in the (i,j) entry if we have observed that entry
    is_observed_matrix_numpy=~np.isnan(observed_matrix)
    theano_observed_matrix.set_value((np.nan_to_num(is_observed_matrix_numpy*observed_matrix)).astype(np.float64))
    is_observed_matrix.set_value(np.nan_to_num(is_observed_matrix_numpy*np.ones((D,N)).astype(np.float64)))
    is_observed_matrix.set_value(is_observed_matrix_numpy.astype(np.float64))
    elbo=ELBO(m_params,v_params, theano_observed_matrix)
    entropy=Entropy(v_params)
    log_joint=LogJ(m_params,v_params, theano_observed_matrix)
    paramt0=time.clock()
    vParamUpdates=lasagne.updates.adadelta(-elbo,v_params)
    paramt1=time.clock()

    
    #%%time
    counter = 0
    elbo_list = []
    keep_updating = True
    if CONSTRUCT_ELBO_LIST==1:
        while keep_updating:
            AdaDeltaStep()
            keep_updating = False if counter>limit else True 
            # if CONSTRUCT_ELBO_LIST==1, estimate ELBO by Monte Carlo every 20 steps
            if counter%20==0:
                elbo_list.append(np.mean([elbo.eval() for i in range(40)]))
            counter +=1
    else:
        while keep_updating:
            AdaDeltaStep()
            keep_updating = False if counter>limit else True 
            counter += 1    
    return elbo_list

#MVNormalScan constructs our estimate of the entire matrix 
def MVNormalScan_BETA0(n, is_observed_matrix, covariance_matrix_mvn_scan, w_mvn_scan, zY, zK, true_full_matrix):
    #construct is_unobserved_matrix a vector of 1s and 0s where the ith coord is a 1 if we haven't seen the ith coord of y_n    
    is_unobserved_matrix=-(is_observed_matrix[:,n]-T.ones(D))
    #construct covariance of the observed entries where the rows/columns with nothing have a 1 on diag (so invertible)
    sigma_observed=T.outer(is_observed_matrix[:,n], is_observed_matrix[:,n])*covariance_matrix_mvn_scan+(is_unobserved_matrix*T.identity_like(covariance_matrix_mvn_scan))
    sigma_unobs_obs=(T.outer(is_unobserved_matrix, is_observed_matrix[:,n]))*covariance_matrix_mvn_scan
    sigma_observed_inv=T.nlinalg.MatrixInverse()(sigma_observed)
    dummy_y=T.zeros(D)
    #draw the mean vector dummy_y from N(0, wwT+sigma^2I) using computationally fast trick
    dummy_results, dummy_updates= theano.scan(lambda prior_result, sigma, zY, w_mvn_scan, zK,n: 
                                              T.sqrt(sigma)*zY[:,n]+T.dot(w_mvn_scan,zK[:,n]) + prior_result,
                                              sequences=None,
                                              outputs_info= T.zeros(D),
                                              non_sequences=[sigma, zY, w_mvn_scan, zK,n],
                                              n_steps=R)
    dummy_y=dummy_results[-1]
    dummy_y /= R
    dummy_y_obs=is_observed_matrix[:,n]*dummy_y
    dummy_y_unobs=is_unobserved_matrix*dummy_y
    y_unobserved= dummy_y_unobs + T.dot(T.dot(sigma_unobs_obs, sigma_observed_inv), (theano_observed_matrix[:,n]-dummy_y_obs))
    y_unobserved=(y_unobserved*is_unobserved_matrix)-(1e6)*is_observed_matrix[:,n]
    return [y_unobserved, sigma_unobs_obs, sigma_observed_inv]

def MVNormalScan(n, is_observed_matrix, covariance_matrix_mvn_scan, w_mvn_scan, zY, zK, true_full_matrix):
    #construct is_unobserved_matrix a vector of 1s and 0s where the ith coord is a 1 if we haven't seen the ith coord of y_n    
    is_unobserved_matrix=-(is_observed_matrix[:,n]-T.ones(D))
    #construct covariance of the observed entries where the rows/columns with nothing have a 1 on diag (so invertible)
    sigma_observed=T.outer(is_observed_matrix[:,n], is_observed_matrix[:,n])*covariance_matrix_mvn_scan+(is_unobserved_matrix*T.identity_like(covariance_matrix_mvn_scan))
    sigma_unobs_obs=(T.outer(is_unobserved_matrix, is_observed_matrix[:,n]))*covariance_matrix_mvn_scan
    sigma_observed_inv=T.nlinalg.MatrixInverse()(sigma_observed)
    dummy_y=T.zeros(D)
    #draw the mean vector dummy_y from N(0, wwT+sigma^2I) using computationally fast trick
    dummy_results, dummy_updates= theano.scan(lambda prior_result, sigma, zY, w_mvn_scan, zK,n: 
                                              T.sqrt(sigma)*zY[:,n]+T.dot(w_mvn_scan,zK[:,n]) + prior_result,
                                              sequences=None,
                                              outputs_info= T.zeros(D),
                                              non_sequences=[sigma, zY, w_mvn_scan, zK,n],
                                              n_steps=R)
    dummy_y=dummy_results[-1]
    dummy_y /= R
    dummy_y_obs=is_observed_matrix[:,n]*dummy_y
    dummy_y_unobs=is_unobserved_matrix*dummy_y
    y_unobserved= dummy_y_unobs + T.dot(T.dot(sigma_unobs_obs, sigma_observed_inv), (theano_observed_matrix[:,n]-dummy_y_obs))
    y_unobserved=(y_unobserved*is_unobserved_matrix)+(true_full_matrix[:,n]+ERROR[0])*is_observed_matrix[:,n]
    #Add true_full_matrix for BETA discount method instead of subtracting infinity from observed entries
    return [y_unobserved, sigma_unobs_obs, sigma_observed_inv]

def MainBandit(observed_matrix, true_full_matrix,ratings, index_list,RANDOM, TRUTH, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list):
    is_observed_matrix_numpy=~np.isnan(observed_matrix)
    theano_observed_matrix.set_value((np.nan_to_num(is_observed_matrix_numpy*observed_matrix)).astype(np.float64))
    is_observed_matrix.set_value(np.nan_to_num(is_observed_matrix_numpy*np.ones((D,N)).astype(np.float64)))
    y_predicted=np.zeros((D,N)) 
    
    #Estimate variance of the variational posterior of wwT+sigmaI w/Monte Carlo
    t1=time.clock()
    Kmean=np.mean([covariance_matrix.eval() for i in range(100)], axis=0)
    Kvar=np.mean([(np.power(covariance_matrix.eval()-Kmean,2)) for i in range(100)], axis=0).sum()
    
    #If the variance is large then run a longer variational inference and observe more entries
    #If variance is small then run a short VI and observe fewer entries
    if (Kvar/(D*D))>0.5:
        limit=1000
        Nobs=int(2*NUM_INITIAL_RANDOM_OBSERVATIONS)
        print(Kvar/(D*D))
        print(Nobs)
    else:
        limit=50
        Nobs=NUM_INITIAL_RANDOM_OBSERVATIONS
    t2=time.clock()
    print("variance check time")
    print(t2-t1)
    
    if RANDOM==1:
        covariance_matrix_mvn_scan=covariance_matrix_random
        w_mvn_scan=srng.uniform(size=(D,K),low=0,high=1000)
    else:
        tG=time.clock()
        elbo_list=GPFA(observed_matrix, limit, CONSTRUCT_ELBO_LIST, elbo_list)
        tG2=time.clock()
        covariance_matrix_mvn_scan=covariance_matrix

        w_mvn_scan=w 

    #draw unobserved entries from multivariate normal conditional on observed entries
    if BETA==0:
        [y_predicted_scan, sigma_u_o_scan, sigma_ob_inv_scan], updates=theano.scan(fn=MVNormalScan_BETA0,
                                              sequences=T.arange(N),
                                              outputs_info=None,
                                              non_sequences=[is_observed_matrix, covariance_matrix_mvn_scan, w_mvn_scan, zY, zK, true_full_matrix])
    else:
        [y_predicted_scan, sigma_u_o_scan, sigma_ob_inv_scan], updates=theano.scan(fn=MVNormalScan,
                                              sequences=T.arange(N),
                                              outputs_info=None,
                                              non_sequences=[is_observed_matrix, covariance_matrix_mvn_scan, w_mvn_scan, zY, zK, true_full_matrix])
    tS2=time.clock()
    y_predicted=np.transpose(y_predicted_scan.eval())
    avgy_predicted=y_predicted
    
    
    if BETA==0:
        observed_indices=[]
        for n in range(Nobs):
            index= np.argmax(y_predicted.flatten())
            i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
            row_list.append(i)
            col_list.append(j)
            rating_error.append(np.abs(np.max(y_predicted.flatten())-true_full_matrix[i,j]))
            observed_indices.append([i,j])
            observed_matrix[i,j]=true_full_matrix[i,j]+ERROR[0].eval()
            y_predicted=np.transpose(y_predicted_scan.eval())
            avgy_predicted+= y_predicted
            for m in range(len(observed_indices)):
                obs_index=observed_indices[m]
                y_predicted[obs_index[0], obs_index[1]]=-(1e6)
            ratings.append(observed_matrix[i,j])
            index_list.append(index)
    else:
        for n in range(Nobs):
            Rewards=np.multiply(y_predicted, np.power(BETA, matrix_sample_count))
            index= np.argmax(Rewards.flatten())
            i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
            reward_raw=true_full_matrix[i,j]+ERROR[0].eval()
            observed_matrix[i,j]=reward_raw
            ratings.append(reward_raw*(BETA**(matrix_sample_count[i,j])))
            index_list.append(index)
            matrix_sample_count[i,j]+=1
            y_predicted=np.transpose(y_predicted_scan.eval())
            avgy_predicted+= y_predicted
    
    avgy_predicted /= (Nobs+1)
    frobenius_error_list.append(((np.multiply(avgy_predicted-true_full_matrix, avgy_predicted-true_full_matrix)).sum())/(np.size(ratings)))
    
    
    return observed_matrix, index_list, ratings, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list


count=0
while count<NUM_ITERATIONS: 
    observed_matrix, index_list, ratings, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list=MainBandit(observed_matrix, true_full_matrix,ratings, index_list,RANDOM, TRUTH,
                                                                                           matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list)
    count +=1
    print(count)
    print('length of ratings')
    print(len(ratings))

#Construct Accumulated Reward and Regret for when entries are picked randomly and when picked optimally
gpfa_accumulated_rewards=np.cumsum(ratings)

best_accumulated_rewards=[]
best_ratings=[]
best_index_list=[]
if BETA==0:
    best_accumulated_rewards=sorted(true_full_matrix.flatten(), reverse=True)
    best_accumulated_rewards=np.cumsum(best_accumulated_rewards)
    random_accumulated_rewards=np.zeros(np.shape(y_comparison))
    RANDOM_RANGE=50
    for r in range(RANDOM_RANGE):
        np.random.shuffle(y_comparison)
        random_accumulated_rewards+=np.cumsum(y_comparison)
    random_accumulated_rewards/=RANDOM_RANGE
else:
    trueRewards=np.multiply(true_full_matrix, np.power(BETA, matrix_sample_count_best))
    for i in range(len(ratings)):
        index= np.argmax(trueRewards.flatten())
        i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
        best_ratings.append(true_full_matrix[i,j]*(BETA**(matrix_sample_count_best[i,j])))
        best_index_list.append(index)
        matrix_sample_count_best[i,j]+=1
        trueRewards=np.multiply(true_full_matrix, np.power(BETA, matrix_sample_count_best))
    best_accumulated_rewards=np.cumsum(best_ratings)
    RANDOM_RANGE=50
    matrix_sample_count_RANDOM_copy=np.zeros((D,N))
    random_accumulated_rewards=np.zeros(len(ratings))
    for r in range(RANDOM_RANGE):
        for d in range(D):
            for n in range(N):
                matrix_sample_count_RANDOM_copy[d,n]=matrix_sample_count_RANDOM[d,n]
        random_ratings=[]
        for i in range(len(ratings)):
            index=np.random.randint(0, high=D*N-1)
            i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
            random_ratings.append(true_full_matrix[i,j]*(BETA**(matrix_sample_count_RANDOM_copy[i,j])))
            matrix_sample_count_RANDOM[i,j]+=1
        random_accumulated_rewards+=np.cumsum(random_ratings)
    random_accumulated_rewards/=RANDOM_RANGE


#EDIT HOW YOU WANT TO SAVE THE DATA. 
#save_path= '/local/data/public/am2648'
#name_of_file='SVI'+'N'+str(N)+'TRUE_RANK'+str(TRUE_RANK)+'seed'+str(seed_number)

#completeName=os.path.join(save_path, name_of_file+'Rewards.csv')
#with file(completeName, 'w') as outfile:
#    np.savetxt(outfile, gpfa_accumulated_rewards, delimiter=',')

#completeName=os.path.join(save_path, name_of_file+'index_list.csv')
#with file(completeName, 'w') as outfile:
#    np.savetxt(outfile, index_list, delimiter=',')

#completeName=os.path.join(save_path, name_of_file+'Frob.csv')
#with file(completeName, 'w') as outfile:
#    np.savetxt(outfile, frobenius_error_list, delimiter=',')

#completeName=os.path.join(save_path, name_of_file+'RatingError.csv')
#with file(completeName, 'w') as outfile:
#    np.savetxt(outfile, rating_error, delimiter=',')

TOTAL_TIME_2=time.clock()
print('TOTAL_TIME')
print(TOTAL_TIME_2-TOTAL_TIME_1)

