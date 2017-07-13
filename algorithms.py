import numpy as np


def s_balcan(buyer):

	#0 indexing instead of 1 indexing as seen in Algo 2 (pg 14) of Balcan et al. WINE 2014
	def compute_aj_by_a0(buyer, j, no_of_item, bit_length, upper_bdd_H, quantity_q=1, extra_amt_xej=0):
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

def s_util_constrained(buyer,realistic=None):
	
	def get_p_ij_bar()

def s_binary_search():
	return NotImplementedError

def s_util_unconstrained():
	return NotImplementedError

if __name__=='__main__':
	from buyers import UtilityBuyer
	np.random.seed(2018)
	no_of_item = 5
	buyer = UtilityBuyer(no_of_item=no_of_item)
	a_estimated = s_balcan(buyer)
	print '\n',a_estimated,buyer.get_valuation_vector()
