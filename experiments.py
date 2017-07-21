import numpy as np
import time, pickle, datetime, copy, collections
import buyers, algorithms, data, geometric



def illustrate_learning_constrained_buyer(params):
	#Local
	no_of_item = params['no_of_item']

	#Model
	buyer = buyers.UtilityBuyer(no_of_item=no_of_item)

	#Algorithm
	final_set0 = algorithms.s_util_constrained(buyer,realistic=None,initial_radius=1.0/2)
	simplex = geometric.Hyperplane(normal=np.ones(no_of_item)*1.0/np.sqrt(no_of_item),rhs=1.0/np.sqrt(no_of_item))
	final_set = geometric.get_ellipsoid_intersect_hyperplane(final_set0,simplex)

	#Logging
	print 'final set 0 center',final_set0.get_center(),'sum',np.sum(final_set0.get_center()), 'final set 0 eigs',final_set0.get_eigenvals()
	print 'final set center',final_set.get_center(),'sum',np.sum(final_set.get_center())
	print 'final set eigs',final_set.get_eigenvals()
	print 'ground truth:',buyer.get_valuation_vector()




if __name__=='__main__':
	np.random.seed(2018)
	params = {}
	params['no_of_item'] = 3
	params['realistic_price_set'] = data.get_realistic_prices_synthetic(params['no_of_item'])

	## Sec 4
	# illustrate_learning_unconstrained_buyer(params)

	## Sec 5
	# illustrate_learning_preference_buyer(params)

	## Sec 6
	illustrate_learning_constrained_buyer(params)

	## Sec 7
	# illustrate_learning_stock_sensitive_buyer(params)