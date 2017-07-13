import numpy as np


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

	def get_volume(self):
		return NotImplementedError

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


if __name__=='__main__':
	np.random.seed(2018)

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
	pvec = np.random.rand(4)
	cvec = np.random.rand(4)
	E0 = Ellipsoid(ctr=cvec,shape_mat=np.eye(pvec.shape[0]))
	H0 = SpecialHalfspace(pvec=pvec,cvec=cvec,direction='leq')
	E1 = get_min_vol_ellipsoid(E0,H0)
	print np.prod(np.linalg.eig(E0.shape_mat)[0]),np.prod(np.linalg.eig(E1.shape_mat)[0])
	H1 = SpecialHalfspace(pvec=np.random.rand(4),cvec=E1.get_center(),direction='geq')
	E2 = get_min_vol_ellipsoid(E1,H1)
	print np.prod(np.linalg.eig(E1.shape_mat)[0]),np.prod(np.linalg.eig(E2.shape_mat)[0])