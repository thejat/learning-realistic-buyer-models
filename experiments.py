import numpy as np
import time, pickle, datetime, copy, collections
import buyers, algorithms, data, geometric



def illustrate_learning_constrained_buyer(params):
	#Locals
	no_of_item = params['no_of_item']

	#Model
	buyer = buyers.UtilityBuyer(no_of_item=no_of_item)

	if 'realistic_price_set' not in params:
		#Algorithm
		final_set0 = algorithms.s_util_constrained(buyer,realistic=None,initial_radius=1.0/2,debug=False)
		simplex = geometric.Hyperplane(normal=np.ones(no_of_item)*1.0/np.sqrt(no_of_item),rhs=1.0/np.sqrt(no_of_item))
		final_set = geometric.get_ellipsoid_intersect_hyperplane(final_set0,simplex)

		#Prior work
		balcan_estimated = algorithms.s_balcan(buyer,debug=False)

	#Logging
	print 'Algorithm 2 of Balcan et al.:\n\t Estimated: ',balcan_estimated,'\tTruth: ',buyer.get_valuation_vector()

	print 'Learn-Budgeted-Utility-Unrealistic:'
	print '\tfinal set 0 center',final_set0.get_center(),'sum',np.sum(final_set0.get_center()), 'final set 0 eigs',final_set0.get_eigenvals()
	print '\tEstimated:',final_set0.get_center()*1.0/np.sum(final_set0.get_center())
	# print '\tEstimated:',final_set.get_center(),'sum',np.sum(final_set.get_center())
	# print '\tfinal set eigs',final_set.get_eigenvals()
	print '\tTruth:',buyer.get_valuation_vector()




if __name__=='__main__':
	np.random.seed(2018)
	params = {}
	params['no_of_item'] = 3

	## Sec 4 (realistic)
	# illustrate_learning_unconstrained_buyer(params)

	## Sec 5 (realistic)
	# illustrate_learning_preference_buyer(params)

	## Sec 6 (unrealistic case)
	illustrate_learning_constrained_buyer(params)

	## Sec 6 (realistic case)
	# params['realistic_price_set'] = data.get_realistic_prices_synthetic(params['no_of_item'])
	# illustrate_learning_constrained_buyer(params)

	## Sec 7 (unrealistic and realistic)
	# illustrate_learning_stock_sensitive_buyer(params)