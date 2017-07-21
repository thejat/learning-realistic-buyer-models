import numpy as np
import pulp #tbd: gurobipy/cplex

class Buyer(object):

	def __init__(self):
		#ideally, this init should not be called
		self.no_of_item = 3
		self.bit_length = 5

	def get_no_of_item(self):
		#assuming this attribute exists, potential bug
		return self.no_of_item

	def get_valuation_vector(self):
		return self.utility_coeffs_linear

	def set_valuation_vector(self):
		self.utility_coeffs_linear = np.zeros(self.no_of_item)
		for k in range(self.no_of_item):
			while self.utility_coeffs_linear[k]==0:
				self.utility_coeffs_linear[k] = np.random.randint(2**self.bit_length)*1.0/(2**self.bit_length)
		self.utility_coeffs_linear = self.utility_coeffs_linear/np.sum(self.utility_coeffs_linear)

class PreferenceBuyer(Buyer):

	def __init__(self,no_of_item=3,bit_length=5):
		self.no_of_item = no_of_item
		self.bit_length = bit_length
		self.set_valuation_vector()
		self.set_buyer_dist()

	def set_buyer_dist(self):

		#below code chooses the support size
		if self.no_of_item < 6:
			self.no_of_types = 1
			for e in range(1,self.no_of_item+1):
				self.no_of_types *= e
		else:
			self.no_of_types = self.no_of_item**3 #hardcoded cubic

		self.types = []
		for e in range(self.no_of_types):
			#assume item indexing is from 0
			self.types.append(np.random.permutation(range(self.no_of_item))) #there could be repetitions here! bug

		self.probabilities = np.random.dirichlet(np.random.randint(1,20,self.no_of_types),1)[0] #hardcoded numbers

	def sample_a_list(self):
		type_index = np.random.choice(range(self.no_of_types), p=self.probabilities)
		# print type_index
		return self.types[type_index]

	def get_buyer_dist(self):
		return (self.types,self.probabilities)

class UtilityBuyer(Buyer):

	def __init__(self,no_of_item=3,bit_length=10):
		self.no_of_item = no_of_item
		self.bit_length = bit_length
		self.eps = 1e-4
		self.set_budget()
		self.set_valuation_vector()

	def get_bit_length(self):
		return self.bit_length

	def get_budget(self):
		return self.budget

	def set_budget(self,budget=None):
		if budget is not None:
			self.budget = budget
		else:
			self.budget = self.no_of_item/5.0 #no_of_item*(np.random.rand()+self.eps)/(1+self.eps)

	def get_constrained_bundle(self,price_vec):
		bundle = pulp.LpVariable.dicts("bundle", range(self.no_of_item), 0, 1) #hardcoded ub, lb
		lp_prob = pulp.LpProblem("Buyer Problem", pulp.LpMaximize)
		dot_ax = pulp.lpSum([self.utility_coeffs_linear[i] * bundle[i] for i in range(self.no_of_item)])
		lp_prob += dot_ax

		dot_px = pulp.lpSum([price_vec[i] * bundle[i] for i in range(self.no_of_item)])
		budget_constraint = dot_px <= self.budget
		lp_prob += budget_constraint

		lp_prob.solve()

		# print "\tConstrained Buyer solve status:", pulp.LpStatus[lp_prob.status]
		t = []
		for v in lp_prob.variables():
			# print "\t",v.name, "=", v.varValue
			t.append(v.varValue)
		# print "\t Obj =", pulp.value(lp_prob.objective)
		return t


	def get_constrained_bundle_non_lp(self,price_vec):
		bang_per_buck = list(self.utility_coeffs_linear/price_vec)

		item_ids_decreasing = [i[0] for i in sorted(enumerate(bang_per_buck), key=lambda x:x[1],reverse=True)] 

		# print bang_per_buck
		# print item_ids_decreasing

		bundle = np.zeros(self.no_of_item)
		left_over_budget = self.get_budget()
		for item_id in item_ids_decreasing:
			if left_over_budget - price_vec[item_id] > 0:
				left_over_budget = left_over_budget - price_vec[item_id]
				bundle[item_id] = 1.0
			else:
				bundle[item_id] = left_over_budget*1.0/price_vec[item_id]
				break

		return bundle


	def get_bundle_given_budget(self,price_vec,budget):
		self.set_budget(budget)
		bundle = self.get_bundle(price_vec)
		return bundle

	def get_bundle(self,price_vec):
		# return self.get_constrained_bundle(price_vec)
		return self.get_constrained_bundle_non_lp(price_vec)


if __name__=='__main__':
	np.random.seed(2018)

	#debugging	
	# no_of_item = 5
	# b = UtilityBuyer(no_of_item=no_of_item)
	# price_vec = np.random.rand(no_of_item)
	# print price_vec
	# print b.get_valuation_vector(),b.get_budget()
	# print b.get_constrained_bundle(price_vec)


	##debugging
	# no_of_item = 5
	# b = UtilityBuyer(no_of_item=no_of_item)
	# price_vec = np.array([0.9,1.02399951e3,1.12589991e15,1.12589991e15,1.12589991e15])
	# print price_vec
	# x = b.get_bundle_given_budget(price_vec,1)
	# print x
	# print np.dot(price_vec,np.array(x))


	##debugging
	b = PreferenceBuyer(no_of_item=3)
	print b.get_valuation_vector()
	types,probabilities = b.get_buyer_dist()
	for e,t in enumerate(types):
		print t, probabilities[e]
	for i in range(10):
		print b.sample_a_list()