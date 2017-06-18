import numpy as np
import pulp #tbd: gurobipy/cplex


class PreferenceBuyer(object):

	def get_buyer_types():
		return NotImplementedError

class UtilityBuyer(object):

	def __init__(self,no_of_item=3,bit_length=20):
		self.no_of_item = no_of_item
		self.bit_length = bit_length
		self.eps = 1e-4
		self.set_budget()
		self.set_valuation_vector()

	def get_valuation_vector(self):
		return self.utility_coeffs_linear

	def set_valuation_vector(self):
		self.utility_coeffs_linear = np.random.randint(2**self.bit_length,size=self.no_of_item)*1.0/(2**self.bit_length)
		self.utility_coeffs_linear = self.utility_coeffs_linear/np.sum(self.utility_coeffs_linear)

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

	def get_bundle_given_budget(self,price_vec,budget):
		self.set_budget(budget)
		bundle = self.get_constrained_bundle(price_vec)
		return bundle


if __name__=='__main__':
	np.random.seed(2018)
	
	no_of_item = 5
	b = UtilityBuyer(no_of_item=no_of_item)
	price_vec = np.random.rand(no_of_item)
	print price_vec

	print b.get_valuation_vector(),b.get_budget()
	print b.get_constrained_bundle(price_vec)
	print b.get_bundle_given_budget(price_vec,b.get_budget())