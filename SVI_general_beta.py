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
BETA=0.1

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
#observation_record[0,n] gives the value of the nth observation, observation_record[1:2,n] give the matrix indices of the observation
observation_record_numpy=np.zeros((3,NUM_OBSERVATIONS_TOTAL))

observed_matrix=np.empty((D,N))
observed_matrix[:]=np.NAN
observed_count=1
y_comparison=[]
matrix_sample_count=np.zeros((D,N))
matrix_sample_count_best=np.zeros((D,N))
matrix_sample_count_random=np.zeros((D,N))
col_list=[]

            

#Define Theano Variables
Shared = lambda shape,name: theano.shared(value = np.ones(shape,dtype=theano.config.floatX),
                                          name=name,borrow=True) 

observation_mask=Shared((D,D), 'observation_mask')
incomplete_identity_matrix=Shared((D,D), "incomplete_identity_matrix")
theano_observed_matrix=Shared((D,N), "theano_observed_matrix")
is_observed_matrix=Shared((D,N), "is_observed_matrix")
#observation_record[0,n] gives the value of the nth observation, observation_record[1:2,n] give the matrix indices of the observation
observation_record=Shared((3,NUM_OBSERVATIONS_TOTAL), "observation_record")

#Define variational parameters
m_matrix_estimate=Shared((D,N), "m_matrix_estimate")
s_matrix_estimate=Shared((D,N), "s_matrix_estimate")
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
#zmatrix_estimate=srng.normal((D,N), "zmatrix_estimate")
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
matrix_estimate=T.exp(m_matrix_estimate+0.1*s_matrix_estimate) #MUST FIX
w=T.exp(m_w+0.1*s_w) #THIS MUST BE FIXED- WHY THE ERROR?
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
v_params= [m_matrix_estimate, s_matrix_estimate, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma]
m_params = [matrix_estimate, w, r, gamma, gamma0, c0, sigma]


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
    m_matrix_estimate, s_matrix_estimate, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma= v_params
    ent_matrix_estimate=(T.log(T.abs_(s_matrix_estimate))+m_matrix_estimate)
    ent_w=(T.log(T.abs_(s_w))+m_w)
    ent_r=(T.log(T.abs_(s_r))+m_r)
    ent_gamma=(T.log(T.abs_(s_gamma))+m_gamma)
    ent_gamma0=(T.log(T.abs_(s_gamma0))+m_gamma0)
    ent_c0=(T.log(T.abs_(s_c0))+m_c0)
    ent_sigma=(T.log(T.abs_(ssigma))+msigma)
    entropy=ent_matrix_estimate.sum()+ent_w.sum()+ent_r.sum()+ent_gamma+ent_gamma0+ent_c0+ent_sigma
    return(entropy)

def LogJ(m_params,v_params,theano_observed_matrix):
    m_matrix_estimate, s_matrix_estimate, m_w, s_w, m_r, s_r, m_gamma, s_gamma, m_gamma0, s_gamma0, m_c0, s_c0, msigma, ssigma=v_params
    matrix_estimate, w, r, gamma, gamma0, c0, sigma= m_params
    is_observed_matrix_numpy=~np.isnan(observed_matrix)
    is_observed_matrix.set_value(is_observed_matrix_numpy.astype(np.float64))
    theano_observed_matrix.set_value((np.nan_to_num(is_observed_matrix_numpy*observed_matrix)).astype(np.float64))
    log_j_t0=time.clock()
    log_joint=0
    for n in range(observed_count):
        log_joint+= np.power((observation_record[0,n]-matrix_estimate[int(observation_record_numpy[1,n]), int(observation_record_numpy[2,n])]),2)
    log_joint=(-1/(2*ERROR[0]*ERROR[0]))*log_joint
    print("first result")
    print(log_joint.eval())
    log_joint += -(N/2.0)*T.nlinalg.Det()(covariance_matrix) - (1/2.0)*T.nlinalg.trace(T.dot(T.dot(matrix_estimate, matrix_estimate.T), T.nlinalg.MatrixInverse()(covariance_matrix)))
    log_joint2= (((D*gamma*T.log(gamma))[0]*r).sum()-(D*T.gammaln(gamma[0]*r)).sum()+((gamma[0]*r-1)*T.log(w)).sum()-(gamma[0]*w).sum() + (gamma0*T.log(c0)-THRESHOLD_RANK*T.gammaln(gamma0/THRESHOLD_RANK)+(gamma0/THRESHOLD_RANK-1)[0]*(T.log(r)).sum()-(c0[0]*r).sum()-gamma-gamma0-c0)[0])
    log_joint += log_joint2
 
    return(log_joint)
    
def ELBO(m_params,v_params, theano_observed_matrix):
    return(LogJ(m_params,v_params, theano_observed_matrix)+Entropy(v_params)[0])

#initialize AdaDeltaStep so that it isn't initialized each time the function is called.
print(theano_observed_matrix.eval())
if RANDOM+TRUTH==0:
    log_joint=LogJ(m_params,v_params, theano_observed_matrix)
    print(log_joint)
    print(log_joint.eval())
    elbo=ELBO(m_params,v_params, theano_observed_matrix)
    print('elbo')
    print(elbo.eval())
    entropy=Entropy(v_params)
    
    paramt0=time.clock()
    vParamUpdates=lasagne.updates.adadelta(-elbo,v_params)
    paramt1=time.clock()
    #AdaDeltaStep=theano.function(inputs=[], updates=vParamUpdates)

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


def MainBandit(observed_matrix, true_full_matrix,ratings, index_list,RANDOM, TRUTH, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list, observed_count):
    is_observed_matrix_numpy=~np.isnan(observed_matrix)
    theano_observed_matrix.set_value((np.nan_to_num(is_observed_matrix_numpy*observed_matrix)).astype(np.float64))
    is_observed_matrix.set_value(np.nan_to_num(is_observed_matrix_numpy*np.ones((D,N)).astype(np.float64)))
    
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
    
    
    for n in range(Nobs):
        Rewards=np.multiply(matrix_estimate.eval(), np.power(BETA, matrix_sample_count))
        index= np.argmax(Rewards.flatten())
        i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
        reward_raw=true_full_matrix[i,j]+ERROR[0].eval()
        observed_matrix[i,j]=reward_raw
        ratings.append(reward_raw*(BETA**(matrix_sample_count[i,j])))
        index_list.append(index)
        observation_record_numpy[0, observed_count]=true_full_matrix[i,j]+ERROR[0].eval()
        observation_record_numpy[1,observed_count]=i
        observation_record_numpy[2,observed_count]=j
        matrix_sample_count[i,j]+=1
        
    observation_record.set_value(observation_record_numpy.astype(np.float64))
    frobenius_error_list.append(((np.multiply(matrix_estimate-true_full_matrix, matrix_estimate-true_full_matrix)).sum())/(np.size(ratings)))
    
    
    return observed_matrix, index_list, ratings, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list, observed_count





#MAIN FUNCTION #################################################################################

#FIRST RANDOMLY SAMPLE SOME ENTRIES
observed_count=0
while observed_count<NUM_INITIAL_RANDOM_OBSERVATIONS:
    d= np.random.randint(0,high=D)
    n=np.random.randint(0, high=int(N/2))
    observed_matrix[d,n]=true_full_matrix[d,n]+ERROR[0].eval()
    observation_record_numpy[0, observed_count]=true_full_matrix[d,n]+ERROR[0].eval()
    observation_record_numpy[1,observed_count]=d
    observation_record_numpy[2,observed_count]=n
    matrix_sample_count[d,n]+=1
    matrix_sample_count_best[d,n]+=1
    matrix_sample_count_random[d,n]+=1
    ratings.append(observed_matrix[d,n]*(BETA**(matrix_sample_count[d,n])))
    index_list.append(Sub2Ind(np.shape(true_full_matrix), d, n))
    observed_count+=1
observation_record.set_value(observation_record_numpy.astype(np.float64))


count=0
while count<NUM_ITERATIONS: 
    observed_matrix, index_list, ratings, matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list, observed_count=MainBandit(observed_matrix, true_full_matrix,ratings, index_list,RANDOM, TRUTH,
                                                                                           matrix_sample_count, elbo_list, frobenius_error_list, rating_error, row_list, col_list, observed_count)
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
    matrix_sample_count_random_copy=np.zeros((D,N))
    random_accumulated_rewards=np.zeros(len(ratings))
    for r in range(RANDOM_RANGE):
        for d in range(D):
            for n in range(N):
                matrix_sample_count_random_copy[d,n]=matrix_sample_count_random[d,n]
        random_ratings=[]
        for i in range(len(ratings)):
            index=np.random.randint(0, high=D*N-1)
            i,j = int(np.floor(index/N)), index-(N*int(np.floor(index/N)))
            random_ratings.append(true_full_matrix[i,j]*(BETA**(matrix_sample_count_random_copy[i,j])))
            matrix_sample_count_random[i,j]+=1
        random_accumulated_rewards+=np.cumsum(random_ratings)
    random_accumulated_rewards/=RANDOM_RANGE


#EDIT HOW YOU WANT TO SAVE THE DATA
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

