import numpy as np 

def get_realistic_prices_bpp():
	return NotImplementedError


def get_realistic_prices_synthetic(no_of_item=3):
	#assume prices are between 0 and 1
	realistic_prices = {}
	for i in range(no_of_item):
		s = np.random.dirichlet(np.random.randint(1,10,3))
		realistic_prices[i] = (np.ceil(s[0]*1e3)*1e-3,np.ceil((s[0]+s[1])*1e3)*1e-3)

	return realistic_prices


if __name__=='__main__':
	np.random.seed(2018)
	print get_realistic_prices_synthetic(3)