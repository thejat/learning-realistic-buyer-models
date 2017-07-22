import numpy as np
import geometric, buyers
np.random.seed(2018)

def s_pref_list(buyer,debug=False):

	types,probabilities = buyer.get_buyer_dist()

	if debug:
		for e,t in enumerate(types):
			print t, probabilities[e]
		
		print '\tA somple type: ', buyer.sample_a_list()

	#TBD

	return 0

def s_util_unconstrained():
	return NotImplementedError

def s_balcan(buyer,debug=False):

	#0 indexing instead of 1 indexing as seen in Algo 2 (pg 14) of Balcan et al. WINE 2014
	def compute_aj_by_a0(buyer, j, no_of_item, bit_length, upper_bdd_H, quantity_q=1, extra_amt_xej=0,debug=False):
		lower_bdd_L = 0
		p = np.zeros(no_of_item)
		p[0] = 1
		for k in range(1,no_of_item):
			p[k] = 2**(10*bit_length)

		i = 0
		flag = None
		while 1:
			i += 1
			p[j] = 0.5*(lower_bdd_L+upper_bdd_H)
			B = extra_amt_xej*p[j] + min(p[0],p[j])*1.0/quantity_q
			x = buyer.get_bundle_given_budget(p,B)
			if debug:
				print 'j=',j,'B=',B,'p[j]=',p[j]
				print '\tp',p,'x=',x
				print '\tlower_bdd_L=',lower_bdd_L,'upper_bdd_H=',upper_bdd_H
			if x[j]>0 and x[0] > 0:
				return p[j]
			if x[j]>0:
				lower_bdd_L = p[j]
				flag = 1
			else:
				# print "lowering H"
				upper_bdd_H = p[j]
				flag = 0

			if i > 4*bit_length:
				break

		return p[j]

		# # tbd
		# if flag==1:
		# 	return np.ceil(p[j])
		# elif flag==0:
		# 	return np.floor(p[j])

	no_of_item = buyer.get_no_of_item()
	bit_length = buyer.get_bit_length()
	a = np.zeros(no_of_item)
	for j in range(1,no_of_item):
		a[j] = compute_aj_by_a0(buyer, j, no_of_item = no_of_item, bit_length=bit_length, upper_bdd_H = 2**(2*bit_length))
	# print a[j]

	a[0] = 1.0/(1+np.sum(a))
	a[1:] = a[1:]*a[0]

	return a

def s_util_constrained(buyer,realistic=None,initial_radius=1.0/2,debug=False):
	'''
	items are zero indexed: 0,1,...,no_of_item-1
	'''

	no_of_item = buyer.get_no_of_item()
	budget = buyer.get_budget() #ideally should be learned as well, see paper.	
	bit_length = buyer.get_bit_length()

	Elist = []
	Elist.append(geometric.Ellipsoid(ctr=buyer.get_valuation_vector()+np.array([0.1,-0.05,0.15]),shape_mat=initial_radius*np.eye(no_of_item)))
	Hlist = [None]
	simplex = geometric.Hyperplane(normal=np.ones(no_of_item)*1.0/np.sqrt(no_of_item),rhs=1.0/np.sqrt(no_of_item))

	for iter_idx in range(1,7):

		#debugging
		if debug:
			print 'iter:',iter_idx,', vol: ',np.around(Elist[iter_idx-1].get_volume(),3) ,',eigs:',np.around(Elist[iter_idx-1].get_eigenvals(),3),',ctr:',np.around(Elist[iter_idx-1].get_center(),3),', a^* belongs:', Elist[iter_idx-1].get_membership(buyer.get_valuation_vector())

		# cylinder_current = geometric.get_ellipsoid_intersect_hyperplane(Elist[iter_idx-1],simplex)

		p_ij_best,p_bar_best,ij_best = get_best_price_from_candidate_prices(no_of_item,budget,bit_length,Elist[iter_idx-1],Elist[iter_idx-1])#,cylinder_current)
		x = buyer.get_bundle(p_ij_best)

		##debugging
		if debug:
			print "\t bundle bought:",x
			print "\t pTa*", np.dot(p_bar_best,buyer.get_valuation_vector())
			print "\t pTc", np.dot(p_bar_best,Elist[iter_idx-1].get_center())


		Hlist.append(get_hyperplane_given_bundle_and_price(x,p_bar_best,ij_best,Elist[iter_idx-1].get_center()))
		if debug:
			print '\t halfspace normal:',Hlist[-1].get_normal(),', rhs:',Hlist[-1].get_rhs()

		temp_ellipsoid = geometric.get_min_vol_ellipsoid(Elist[iter_idx-1],Hlist[iter_idx])
		if temp_ellipsoid.get_membership(buyer.get_valuation_vector()) is False:
			break

		Elist.append(temp_ellipsoid)

		#debugging
		if debug:
			geometric.plot_debug(Elist[-2:],hyperplane=Hlist[-1],halfspace=Hlist[-1],custom_point=buyer.get_valuation_vector())

		

	# H0 = SpecialHalfspace(pvec=pvec,cvec=cvec,direction='leq')
	# E1 = get_min_vol_ellipsoid(E0,H0)
	# print np.prod(np.linalg.eig(E0.shape_mat)[0]),np.prod(np.linalg.eig(E1.shape_mat)[0])
	# H1 = SpecialHalfspace(pvec=np.random.rand(4),cvec=E1.get_center(),direction='geq')
	# E2 = get_min_vol_ellipsoid(E1,H1)
	# print np.prod(np.linalg.eig(E1.shape_mat)[0]),np.prod(np.linalg.eig(E2.shape_mat)[0])

	return Elist[-1]

def get_hyperplane_given_bundle_and_price(x,p_bar_best,ij_best,c):
	'''
	We know only i,j should be sensitized
	'''

	# print "pTc", np.dot(p_bar_best,c)
	tolerance = 1e-5
	assert abs(np.dot(p_bar_best,c)) <= tolerance


	if (x[ij_best[0]] > 0 and x[ij_best[1]] == 0) or (x[ij_best[0]] == 1 and x[ij_best[1]] < 1):
		#then a1^*/p1 \geq a2^*/p2 or np.dot(p_bar_best,a^*) geq 0
		H = geometric.SpecialHalfspace(pvec=p_bar_best,cvec=c,direction='geq')
	else:
		# a1/p1 \leq a2/p2 or np.dot(p_bar_best,a^*) leq 0
		H = geometric.SpecialHalfspace(pvec=p_bar_best,cvec=c,direction='leq')

	return H

def get_best_price_from_candidate_prices(no_of_item,budget,bit_length,ellipsoid,objective):

	c = ellipsoid.get_center()

	ij_best = (0,1)
	quad_val_best = 0
	p_bar_best = np.zeros(no_of_item)
	p_candidate_dict = {}
	for i in range(no_of_item):
		for j in range(no_of_item):
			if i==j:
				continue

			p_candidate_dict[(i,j)] = budget*(2**bit_length)*np.ones(no_of_item)
			p_candidate_dict[(i,j)][i] = budget
			p_candidate_dict[(i,j)][j] = budget*c[j]*1.0/c[i]

			p_bar_ij = np.zeros(no_of_item)
			p_bar_ij[i] = p_candidate_dict[(i,j)][j]
			p_bar_ij[j] = -p_candidate_dict[(i,j)][i]
			p_bar_ij = p_bar_ij*1.0/np.linalg.norm(p_bar_ij,2)

			quad_val_ij = np.dot(p_bar_ij,np.dot(objective.shape_mat,p_bar_ij))

			##debugging
			# print i,j,': ',quad_val_ij
			
			if  quad_val_ij > quad_val_best:
				quad_val_best = quad_val_ij
				ij_best = (i,j)
				p_bar_best = p_bar_ij
	# print ij_best

	return [p_candidate_dict[ij_best],p_bar_best,ij_best]



if __name__=='__main__':
	np.random.seed(2018)



	##debugging s_pref_list
	no_of_item = 10
	buyer = buyers.PreferenceBuyer(no_of_item=no_of_item)
	a_estimated_pref_list = s_pref_list(buyer,debug=True)
	print 'Estimated: ',a_estimated_pref_list,' Truth: ',buyer.get_valuation_vector()


	##debugging s_balcan
	# no_of_item = 3
	# buyer = buyers.UtilityBuyer(no_of_item=no_of_item)
	# a_estimated_balcan = s_balcan(buyer)
	# print '\n',a_estimated_balcan,buyer.get_valuation_vector()

	##debugging s_util_constrained 1
	# no_of_item = 3
	# buyer = buyers.UtilityBuyer(no_of_item=no_of_item)
	# final_set0 = s_util_constrained(buyer,realistic=None,initial_radius=1.0/2)
	# print 'final set 0 center',final_set0.get_center(),'sum',np.sum(final_set0.get_center()), 'final set 0 eigs',final_set0.get_eigenvals()
	# simplex = geometric.Hyperplane(normal=np.ones(no_of_item)*1.0/np.sqrt(no_of_item),rhs=1.0/np.sqrt(no_of_item))
	# final_set = geometric.get_ellipsoid_intersect_hyperplane(final_set0,simplex)
	# print 'final set center',final_set.get_center(),'sum',np.sum(final_set.get_center())
	# print 'final set eigs',final_set.get_eigenvals()
	# print 'ground truth:',buyer.get_valuation_vector()

	##debugging s_util_constrained 2
	# no_of_item = 3
	# buyer = buyers.UtilityBuyer(no_of_item=no_of_item)
	# no_of_item = buyer.get_no_of_item()
	# budget = buyer.get_budget()
	# bit_length = buyer.get_bit_length()
	# Elist = []
	# Elist.append(geometric.Ellipsoid(ctr=buyer.get_valuation_vector(),shape_mat=1.0/2*np.eye(no_of_item)))
	# p_ij_best,p_bar_best,ij_best = get_best_price_from_candidate_prices(no_of_item,budget,bit_length,Elist[0])
	# x = buyer.get_bundle(p_ij_best)
	# Hlist = [None]
	# iter_idx = 1
	# Hlist.append(get_hyperplane_given_bundle_and_price(x,p_bar_best,ij_best,Elist[iter_idx-1].get_center()))
	# Elist.append(geometric.get_min_vol_ellipsoid(Elist[iter_idx-1],Hlist[iter_idx]))
	# p_ij_best1,p_bar_best1,ij_best1 = get_best_price_from_candidate_prices(no_of_item,budget,bit_length,Elist[1])
