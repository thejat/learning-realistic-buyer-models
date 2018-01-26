import numpy as np
import time, pickle, datetime, copy, collections
import buyers, algorithms, data, geometric
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def illustrate_learning_preference_buyer(params):
	# Local
	no_of_item = params['no_of_item']

	# Model
	buyer = buyers.PreferenceBuyer(no_of_item=no_of_item)

	estimated_valuation_ranges = algorithms.s_pref_list(buyer, delta = 0.00001, debug=False)
	print 'estimated valuation ranges:', estimated_valuation_ranges
	print 'true valuation vector:', buyer.get_valuation_vector()

def illustrate_learning_constrained_buyer(params):
	# Local
	no_of_item = params['no_of_item']

	# Model
	buyer = buyers.UtilityBuyer(no_of_item=no_of_item)

	if 'realistic_price_set' not in params:
		# Algorithm
		final_set0 = algorithms.s_util_constrained(buyer, realistic=None, initial_radius=1.0, debug=True)
		simplex = geometric.Hyperplane(normal=np.ones(no_of_item) * 1.0 / np.sqrt(no_of_item), rhs=1.0 / np.sqrt(no_of_item))
		final_set = geometric.get_ellipsoid_intersect_hyperplane(final_set0, simplex)

		# Prior work
		# balcan_estimated = algorithms.s_balcan(buyer, debug=False)

	# Logging
	# print 'Algorithm 2 of Balcan et al.:\n\t Estimated: ', balcan_estimated, '\tTruth: ', buyer.get_valuation_vector()

	# print 'Learn-Budgeted-Utility-Unrealistic:'
	# print '\tfinal set 0 center', final_set0.get_center(), 'sum', np.sum(final_set0.get_center()), 'final set 0 eigs', final_set0.get_eigenvals()
	# print '\tEstimated:', final_set0.get_center() * 1.0 / np.sum(final_set0.get_center())
	# # print '\tEstimated:',final_set.get_center(),'sum',np.sum(final_set.get_center())
	# # print '\tfinal set eigs',final_set.get_eigenvals()
	# print '\tTruth:', buyer.get_valuation_vector()

def illustrate_learning_unconstrained_buyer(params):
	
	number_of_simulations = 3
	number_of_iterations_per_simulation = 6
	plot_matrix = np.zeros((number_of_simulations,number_of_iterations_per_simulation)) # number of different a's times numer of iterations

	no_of_item = 2

	for i in range(number_of_simulations):
		buyer = buyers.Buyer(no_of_item=no_of_item)
		print "valuation = ", buyer.get_valuation_vector()
		plot_matrix[i] = algorithms.s_util_unconstrained(number_of_iterations_per_simulation, buyer, epsilon=0.01)	
	#print plot_matrix[0]	
	get_plot_subroutine(plot_matrix)
	
def get_plot_subroutine(plot_matrix):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	xs = np.arange(len(plot_matrix[0]))
	print xs
	# ys_lb = np.asarray(   [np.percentile(plot_matrix[i,:],25)  for i in range(len(xs)) ]   )
	# ys_ub = np.asarray(   [np.percentile(plot_matrix[i,:],75)  for i in range(len(xs)) ]   )
	# ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	ys = np.asarray(  [np.mean(plot_matrix[i]) for i in range(len(xs))]    )
	ax.plot(xs,ys)
	plt.show()		

if __name__ == '__main__':
	np.random.seed(2018)
	params = {}
	params['no_of_item'] = 5

	# # Sec 4 (realistic)
	# illustrate_learning_unconstrained_buyer(params) #####
	#illustrate_learning_unconstrained_buyer(params)

	# # Sec 5 (realistic)
	# illustrate_learning_preference_buyer(params)

	# # Sec 6 (unrealistic case)
	#illustrate_learning_constrained_buyer(params)

	# # Sec 6 (realistic case)
	# params['realistic_price_set'] = data.get_realistic_prices_synthetic(params['no_of_item'])
	# illustrate_learning_constrained_buyer(params)

	# # Sec 7 (unrealistic and realistic)
	# illustrate_learning_stock_sensitive_buyer(params)





	number_of_simulations = 1
	number_of_iterations_per_simulation = 30
	plot_matrix = np.zeros((number_of_simulations,number_of_iterations_per_simulation)) # number of different a's times numer of iterations

	no_of_item = 2

	for i in range(number_of_simulations):
		buyer = buyers.Buyer(no_of_item=no_of_item)
		print "valuation = ", buyer.get_valuation_vector()
		plot_matrix[i] = algorithms.s_util_unconstrained(number_of_iterations_per_simulation, buyer, epsilon=0.01)	


	fig = plt.figure()
	ax = fig.add_subplot(111)
	xs = np.arange(len(plot_matrix[0]))
	print xs
	ys_lb = np.asarray(   [np.percentile(plot_matrix[:,i],25)  for i in range(len(xs)) ]   )
	ys_ub = np.asarray(   [np.percentile(plot_matrix[:,i],75)  for i in range(len(xs)) ]   )
	ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	ys = np.asarray(  [np.mean(plot_matrix[:,i]) for i in range(len(xs))]    )
	ax.plot(xs,ys)
	plt.show()