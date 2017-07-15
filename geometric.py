import numpy as np
from scipy.special import gamma

class Ellipsoid(object):
	'''
	E(shape_mat,ctr) = (x-ctr)' (shape_mat)^{-1} (x-ctr) \leq 1

	vol(E) = Vn* (\prod(eigs(A)))^{1/dim}

	'''

	def __init__(self,ctr=None,shape_mat=None):
		if ctr is not None:
			self.set_center(ctr)
		if shape_mat is not None:
			self.set_shape_mat(shape_mat)

	def set_center(self,new_ctr):
		self.ctr = new_ctr

	def get_center(self):
		return self.ctr

	def set_shape_mat(self,new_shape_mat):
		self.shape_mat = new_shape_mat

	def get_shape_mat(self):
		return self.shape_mat

	def get_dimension(self):
		return self.ctr.shape[0]

	def get_eigenvals(self):
		return np.linalg.eig(self.get_shape_mat())[0]

	def get_volume(self):
		prop_const = (np.pi**(0.5*self.get_dimension()))*1.0/gamma(1 + 0.5*self.get_dimension()) #https://en.wikipedia.org/wiki/Volume_of_an_n-ball
		return prop_const*np.sqrt(np.prod(self.get_eigenvals()))

	def get_membership(self,point):
		delta = point - self.get_center()
		if np.dot(delta,np.dot(self.get_shape_mat(),delta)) <= 1:
			return True
		else:
			return False

class Halfspace(object):
	'''
	{ x: normal'x \leq rhs} is the general expression

	'''

	def __init__(self,normal=None,rhs=None):
		if normal is not None:
			self.set_normal(normal)
		if rhs is not None:
			self.set_rhs(rhs)

	def set_normal(self,normal):
		self.normal=normal

	def get_normal(self):
		return self.normal

	def set_rhs(self,rhs):
		self.rhs = rhs

	def get_rhs(self):
		return self.rhs

	def get_dimension(self):
		return self.normal.shape[0]

class Hyperplane(Halfspace):
	def is_hyperplane():
		return True

class SpecialHalfspace(Halfspace):
	'''
	special halfspaces
	1. {x: pvec'x \geq pvec'cvec }

	2. {x: pvec'x \leq pvec'cvec }

	'''
	def __init__(self,pvec=None,cvec=None,direction='leq'):

		self.set_direction(direction)

		if pvec is not None and cvec is not None:
			self.set_pvec(pvec)
			self.set_cvec(cvec)
			rhs = np.dot(cvec,pvec)
			normal = pvec
			if direction=='geq':
				normal = -normal
				rhs = -rhs
			super(SpecialHalfspace, self).__init__(normal,rhs)

	def set_direction(self,direction):
		'''
		only two strings: leq and geq
		'''
		self.direction = direction

	def get_direction(self):
		return self.direction

	def set_cvec(self,cvec):
		self.cvec = cvec

	def set_pvec(self,pvec):
		self.pvec = pvec

	def get_pvec(self):
		return self.pvec

def get_min_vol_ellipsoid(ellipsoid,halfpsace):
	tolerance = 1e-4

	if ellipsoid.get_dimension() != halfpsace.get_dimension():
		print "Dimensional error!"
		return Ellipsoid()

	n = ellipsoid.get_dimension()
	A = ellipsoid.get_shape_mat()
	c = ellipsoid.get_center()
	p = halfpsace.get_pvec()

	#a useless check
	if abs(abs(np.dot(c,p))-abs(halfpsace.get_rhs()))>tolerance:
		print "Hyperplane does not pass through center",abs(np.dot(c,p)),abs(halfpsace.get_rhs())
		return Ellipsoid()


	b = np.dot(A,p)*1.0/np.sqrt(np.dot(p,np.dot(A,p)))

	if halfpsace.get_direction()=='leq':
		new_c = c - b*1.0/(n+1)
	else:
		new_c = c + b*1.0/(n+1)

	new_A = ((n**2)*1.0/((n**2)-1))*(A - (2.0/(n+1))*np.outer(b,b))

	return Ellipsoid(ctr=new_c,shape_mat=new_A)

def get_ellipsoid_intersect_hyperplane(ellipsoid,hyperplane):
	# https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf page 13

	def get_svd_of_vector(a0):

		d = a0.shape[0]
		a = a0*1.0/np.linalg.norm(a0,2)
		vecs = np.eye(d)[1:] #hardcoded choice, will cause bugs!
		for i in range(d-1):
			vecs[i] = vecs[i] - np.dot(vecs[i],a)*a
		vecs2,_ = np.linalg.qr(vecs.transpose()) #gram schmidt of the columns
		orthonormal1 = np.hstack((a[:,None],vecs2))

		##debugging
		# test1 = np.dot(orthonormal1,orthonormal1.transpose())
		# eigs1 = np.linalg.eig(test1)[0]
		# test2 = np.dot(orthonormal1.transpose(),orthonormal1)
		# eigs2 = np.linalg.eig(test1)[0]

		sigmavec = np.zeros(d)
		sigmavec[0] = np.linalg.norm(a0,2)

		##debugging
		# print np.dot(orthonormal1,sigmavec)

		return orthonormal1,sigmavec,1


	c = hyperplane.get_normal()
	gamma = hyperplane.get_rhs()
	q = ellipsoid.get_center()
	Q = ellipsoid.get_shape_mat()

	Uv,Sigv,_ = get_svd_of_vector(c)
	w = np.zeros(ellipsoid.get_dimension())
	w[0] = 1
	Uw,Sigw,_ = get_svd_of_vector(w)
	S = np.dot(Uw,Uv.transpose())

	q_prime = q - gamma*np.dot(S,c)*1.0/np.linalg.norm(c)
	Q_prime = np.dot(S,np.dot(Q,S.transpose()))

	M = np.linalg.inv(Q_prime)
	m11 = M[0,0]
	m_bar = M[0,1:]
	M_bar = M[1:,1:]
	M_bar_inv = np.linalg.inv(M_bar)

	w_prime = q_prime + q_prime[0]*np.hstack([-1,np.dot(M_bar_inv,m_bar)])

	W_prime = 0*np.copy(M)
	W_prime[1:,1:] = M_bar_inv
	W_prime = (1 - (q_prime[0]**2)*(m11 - np.dot(m_bar,np.dot(M_bar_inv,m_bar))))*W_prime

	w = np.dot(S.transpose(),w_prime) + gamma*c*1.0/np.linalg.norm(c,2)
	W = np.dot(S.transpose(),np.dot(W_prime,S))

	output_ellipsoid = Ellipsoid(ctr=w,shape_mat=W)

	return output_ellipsoid



if __name__=='__main__':
	np.random.seed(2018)
	np.set_printoptions(precision=4)

	##debugging
	# E1 = Ellipsoid(ctr=np.array([0,0]),shape_mat=np.array([[1,0],[0,1]]))
	# print E1.get_shape_mat()
	# print E1.get_center()

	##debugging
	# pvec = np.random.rand(4)
	# cvec = np.random.rand(4)
	# H1 = SpecialHalfspace(pvec=pvec,cvec=cvec,direction='geq')
	# H2 = SpecialHalfspace(pvec=pvec,cvec=cvec,direction='leq')

	##debugging
	# pvec = np.random.rand(4)
	# cvec = np.random.rand(4)
	# E0 = Ellipsoid(ctr=cvec,shape_mat=np.eye(pvec.shape[0]))
	# H0 = SpecialHalfspace(pvec=pvec,cvec=cvec,direction='leq')
	# E1 = get_min_vol_ellipsoid(E0,H0)
	# print np.prod(np.linalg.eig(E0.shape_mat)[0]),np.prod(np.linalg.eig(E1.shape_mat)[0])
	# H1 = SpecialHalfspace(pvec=np.random.rand(4),cvec=E1.get_center(),direction='geq')
	# E2 = get_min_vol_ellipsoid(E1,H1)
	# print np.prod(np.linalg.eig(E1.shape_mat)[0]),np.prod(np.linalg.eig(E2.shape_mat)[0])



	##debugging
	# a0 = np.random.rand(5)
	# u,s,v = get_svd_of_vector(a0)


	##debugging
	# dim = 3
	# cvec = np.random.rand(dim)
	# for i in range(dim):
	# 	if cvec[i] <0:
	# 		cvec[i] = -cvec[i]
	# cvec = cvec*1.1/np.linalg.norm(cvec,2)
	# ellipsoid = Ellipsoid(ctr=cvec,shape_mat=np.eye(cvec.shape[0]))
	# hyperplane = Hyperplane(normal=np.ones(dim),rhs=1)
	# output_ellipsoid = get_ellipsoid_intersect_hyperplane(ellipsoid,hyperplane)

	dim = 3
	ellipsoid = Ellipsoid(ctr=np.zeros(dim),shape_mat=np.eye(dim))
	hyperplane = Hyperplane(normal=np.ones(dim),rhs=1)
	output_ellipsoid = get_ellipsoid_intersect_hyperplane(ellipsoid,hyperplane)
	#np.ones(dim)*1.0/dim