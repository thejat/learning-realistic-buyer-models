import numpy as np
import time, pickle, datetime, copy, collections
from buyer_models import UtilityBuyer, PreferenceBuyer
from data import get_bpp_price_ranges, get_synthetic_price_ranges
from algorithms import s_binary_search,s_util_unconstrained, s_util_constrained, s_balcan




def exp_learn_util_params():
	#parameters required
	np.random.seed(1000)
	price_range = 1000      #denotes highest possible price of a product
	eps         = 0.1       #tolerance
	N           = 30 #   #number of times Monte Carlo simulation will run
	prodList    = [100, 250, 500, 1000, 3000, 5000, 7000,10000,20000]
	algos = collections.OrderedDict({'Assort-Exact':capAst_AssortExact,'Assort-LSH':capAst_AssortLSH,'Adxopt':capAst_adxopt,'LP':capAst_LP})#,'Static-MNL':capAst_paat}
benchmark = 'LP'#'Static-MNL'#
loggs = get_log_dict(prodList,N,algos,price_range,eps,C)

else:
prodList    = [100,200,400,800,1600]
algos       = collections.OrderedDict({'Linear-Search':genAst_oracle,'Assort-Exact-G':genAst_AssortExact,'Assort-LSH-G':genAst_AssortLSH})
benchmark   = 'Linear-Search'
loggs = get_log_dict(prodList,N,algos,price_range,eps)
loggs['additional']['lenFeasibles'] = np.zeros(len(prodList))


badError = 0
t1= time.time()
for i,prod in enumerate(prodList):

t0 = time.time()
t = 0
while(t<N):

print 'Iteration number is ', str(t+1),' of ',N,', for prod size ',prod

#generating the price
meta = {'eps':eps}
if flag_capacitated == True:
p,v = generate_instance(price_range,prod,genMethod,t)
else:
p,v,feasibles,C,prod = generate_instance_general(price_range,prod,genMethod,t)
loggs['additional']['C'][i,t] = C
meta['feasibles'] = feasibles

#preprocessing for proposed algos
if 'Assort-Exact' in algos:
meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'special_case_exact')
if 'Assort-LSH' in algos:
meta['db_LSH'],_,_ = preprocess(prod, C, p, 'special_case_LSH', nEst=20,nCand=80)#Hardcoded values
if 'Assort-Exact-G' in algos:
meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
if 'Assort-LSH-G' in algos:
meta['db_LSH'],_,_ = preprocess(prod, C, p, 'general_case_LSH', nEst=20,nCand=80,feasibles=feasibles)#Hardcoded values



#run algos
maxSetBenchmark = None
for algoname in algos:
print '\tExecuting ',algoname
loggs[algoname]['rev'][i,t],loggs[algoname]['maxSet'][(i,t)],loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
print '\t\tTime taken is ',loggs[algoname]['time'][i,t],'sec.'

if algoname==benchmark:
maxSetBenchmark = copy.deepcopy(loggs[algoname]['maxSet'][(i,t)])

loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps)

t = t+1    



print 'Experiments (',N,' sims) for number of products ',prod, ' is done.'  
print 'Cumulative time taken is', time.time() - t0,'\n'   
loggs = compute_summary_stats(algos,loggs,benchmark,i)
if flag_capacitated != True:
loggs['additional']['lenFeasibles'][i] = len(feasibles)

#dump it incrementally for each product size
if flag_savedata == True:
if flag_capacitated == True:
pickle.dump(loggs,open('./output/cap_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))
else:
pickle.dump(loggs,open('./output/gen_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
print "Summary:"
for algoname in algos:
print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

return loggs





if __name__=='__main__':

	loggs = exp_learn_util_params(flag_savedata = True, data_type = 'synthetic')






######




def get_real_prices(price_range, prod, iterNum = 0):
  fname = os.getcwd() + '/billion_price_data/processed_data/usa_2/numProducts_stats.npz'
  dateList = np.load(fname)['good_dates']
  fileName = os.getcwd() + '/billion_price_data/processed_data/usa_2/prices_'
  fileNameList = []
  for chosenDay in dateList:
    fileNameList.append(fileName+ chosenDay+'.npz')

  allPrices = np.load(fileNameList[iterNum])['arr_0']
  allPrices = allPrices[np.isfinite(allPrices)]
  # print allPrices
  allValidPrices = allPrices[allPrices > 0.01]
  allValidPrices = allValidPrices[allValidPrices < price_range]
  allValidPrices = sorted(list(allValidPrices))
  p = allValidPrices[:prod]
  # p = random.sample(allValidPrices, prod)
  return p 

def generate_instance(price_range,prod,genMethod,iterNum):
  if genMethod=='bppData':
    p = get_real_prices(price_range, prod, iterNum)
  else:
    p = price_range * np.random.beta(1,1,prod) 
  p = np.around(p, decimals =2)
  p = np.insert(p,0,0) #inserting 0 as the first element to denote the price of the no purchase option
  
  #generating the customer preference vector, we don't care that it is in 0,1. Want it away from 0 for numeric. stability.
  v = np.around(np.random.rand(prod+1) + 1e-3, decimals =7) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
  #Ensure that there are no duplicate entires in v - required for Static-MNL.
  u, indices = np.unique(v, return_inverse=True)   

  while(not(len(u)== prod+1) or abs(v[0])<1e-3):
      if abs(v[0])<1e-3:
        v[0] = np.around(np.random.rand(1) + 1e-3,decimals =7)
        u, indices = np.unique(v, return_inverse=True) 
      extraSize = prod+1 - len(u)
      newEnt = np.around(np.random.rand(extraSize)+1e-3,decimals=7)
      v= np.concatenate((u,newEnt))
      u, indices = np.unique(v, return_inverse=True)

  return p,np.around(v,decimals=7)

def get_log_dict(prodList,N,algos,price_range,eps,C=None):

  def matrices(prodList,N):
    names1 = ['revPctErr','setOlp','corrSet','rev','time']
    names2 = ['corrSet_mean', 'setOlp_mean',  'revPctErr_max', 'revPctErr_mean','revPctErr_std', 'time_mean', 'time_std'] 
    output = {}
    for name in names1:
     output[name] = np.zeros((len(prodList), N))
    for name in names2: 
      output[name] = np.zeros(len(prodList)) 
    return output

  loggs = collections.OrderedDict()
  loggs['additional'] = {'prodList':prodList,'algonames':algos.keys(),'N':N,'eps':eps,'price_range':price_range}
  if C is not None:
    loggs['additional']['C'] = C
  else:
    loggs['additional']['C'] = np.zeros((len(prodList), N))

  for algoname in algos:
    loggs[algoname] = matrices(prodList,N)

    loggs[algoname]['maxSet'] = {}

  return loggs

def compute_summary_stats(algos,loggs,benchmark,i):
  for algoname in algos:
    # print algoname
    if benchmark in algos:
      loggs[algoname]['revPctErr'][i] = (loggs[benchmark]['rev'][i,:] - loggs[algoname]['rev'][i,:])/(loggs[benchmark]['rev'][i,:]+1e-6)
      loggs[algoname]['revPctErr_mean'][i] = np.mean(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_std'][i] = np.std(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_max'][i] = np.max(loggs[algoname]['revPctErr'][i,:])
    loggs[algoname]['corrSet_mean'][i] = np.mean(loggs[algoname]['corrSet'][i,:])
    loggs[algoname]['setOlp_mean'][i] = np.mean(loggs[algoname]['setOlp'][i,:])
    loggs[algoname]['time_mean'][i] = np.mean(loggs[algoname]['time'][i,:])
    loggs[algoname]['time_std'][i] = np.std(loggs[algoname]['time'][i,:])

  return loggs

def compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps):

  def overlap(maxSet,maxSetBenchmark):
    setOlp  = len(maxSetBenchmark.intersection(maxSet))
    corrSet = int(setOlp==  len(maxSetBenchmark))
    setOlp  = setOlp*1.0/len(maxSetBenchmark) #to normalize
    return setOlp,corrSet

  if benchmark in algos:
    for algoname in algos:
      # print 'Collecting benchmarks for ',algoname
      loggs[algoname]['setOlp'][i,t],loggs[algoname]['corrSet'][i,t] = overlap(loggs[algoname]['maxSet'][(i,t)],maxSetBenchmark)
      if(loggs[benchmark]['rev'][i,t] - loggs[algoname]['rev'][i,t] > eps ):
          badError = badError +1
  return loggs,badError

def generate_instance_general(price_range,prod,genMethod,iterNum,lenFeas=None,real_data=None):

  #arbitrary sets

  if real_data is None:
    if lenFeas is None:
      nsets = int(prod**1.5)
    else:
      nsets = lenFeas

    #synthetic
    feasibles = []
    C = 0
    for i in range(nsets):
      temp = random.randint(1,2**prod-1)
      temp2 = [int(x) for x in format(temp,'0'+str(prod)+'b')]
      set_char_vector = np.asarray(temp2)
      feasibles.append(set_char_vector)
      C = max(C,np.sum(set_char_vector))
  else:
    #real
    feasibles,C,prod = get_feasibles_realdata(fname=real_data['fname'],isCSV=real_data['isCSV'],min_ast_length=real_data['min_ast_length'])


  p,v = generate_instance(price_range,prod,genMethod,iterNum)


  return p,v,feasibles,int(C),prod



