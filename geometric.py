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
		return self.get_center().shape[0]

	def get_eigenvals(self):
		return np.linalg.eig(self.get_shape_mat())[0]

	def get_volume(self):
		prop_const = (np.pi**(0.5*self.get_dimension()))*1.0/gamma(1 + 0.5*self.get_dimension()) #https://en.wikipedia.org/wiki/Volume_of_an_n-ball
		return prop_const*np.sqrt(np.prod(self.get_eigenvals()))

	def get_membership(self,point):
		delta = point - self.get_center()
		if np.dot(delta,np.dot(np.linalg.inv(self.get_shape_mat()),delta)) <= 1:
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
	def __init__(self,pvec=None,cvec=None,direction='leq',rhs=None):

		self.set_direction(direction)

		if pvec is not None and cvec is not None:
			self.set_pvec(pvec)
			self.set_cvec(cvec)
			if rhs is None:
				rhs = np.dot(cvec,pvec)
			normal = pvec
			# if direction=='geq':
			# 	normal = -normal
			# 	rhs = -rhs
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

	# #a useless check
	# if abs(abs(np.dot(c,p))-abs(halfpsace.get_rhs()))>tolerance:
	# 	print "Hyperplane does not pass through origin",abs(np.dot(c,p)),abs(halfpsace.get_rhs())
	# 	return Ellipsoid()


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

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_debug(ellipsoids,hyperplane=None,halfspace=None,custom_point=None):

	fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
	ax = fig.add_subplot(111, projection='3d')

	# Set of all spherical angles:
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	max_radius = 0.1
	clr = ['b','r']

	for ti,ellipsoid in enumerate(ellipsoids):

		rx, ry, rz = ellipsoid.get_eigenvals()
		ctr = ellipsoid.get_center()
		# Cartesian coordinates that correspond to the spherical angles:
		# (this is the equation of an ellipsoid):
		x = rx * np.outer(np.cos(u), np.sin(v)) 
		y = ry * np.outer(np.sin(u), np.sin(v))
		z = rz * np.outer(np.ones_like(u), np.cos(v))

		# Plot:
		ax.plot_surface(x + ctr[0], y + ctr[1], z + ctr[2],  rstride=4, cstride=4, color=clr[ti],alpha=0.2)
		ax.scatter(ctr[0],ctr[1],ctr[2],color=clr[ti],s=30)
		# ctr = np.around(ctr,3) 
		# ax.text(ctr[0],ctr[1],ctr[2],  '%s,%s,%s' % (str(ctr[0]),str(ctr[1]),str(ctr[2])), size=20, zorder=1, color='k')

		# Adjustment of the axes, so that they all have the same span:
		max_radius = max(max_radius,np.max(abs(np.array([rx, ry, rz])) + abs(ctr)))

	if hyperplane is not None:
		hx = np.linspace(-max_radius,max_radius,5)
		hy = np.linspace(-max_radius,max_radius,5)
		hx,hy = np.meshgrid(hx,hy)
		hc = hyperplane.get_normal()
		hz = (hyperplane.get_rhs() - hc[0]*hx - hc[1]*hy)*1.0/(hc[2]+1e-4)
		ax.plot_surface(hx, hy, hz,rstride=4, cstride=4, color='r',alpha=0.2)

	if halfspace is not None:
		hx = np.linspace(-max_radius,max_radius,5)
		hy = hx
		hz = hx
		hc = halfspace.get_normal()
		pts = []
		for i in hx:
			for j in hy:
				for k in hz:
					if (hc[0]*i + hc[1]*j + hc[2]*k <= halfspace.get_rhs()):
						ax.scatter(i,j,k,color='g',marker='.',alpha=0.5,s=20)

	if custom_point is not None:
		ctr = np.around(custom_point,3)
		ax.scatter(ctr[0],ctr[1],ctr[2],color='b',marker='*',s=30) 
		# ax.text(ctr[0],ctr[1],ctr[2],  'true: %s,%s,%s' % (str(ctr[0]),str(ctr[1]),str(ctr[2])), size=20, zorder=1, color='k')

	for axis in 'xyz':
		getattr(ax, 'set_{}lim'.format(axis))((0, max_radius))

	plt.show()

#############################

from scipy.stats import chi2
def plot_debug2D(ellipsoids, halfspace=None, custom_point=None, halfspace2 = None, theta_num=1e3,ax=None,plot_kwargs=None, fill=False,fill_kwargs=None,mass_level=0.68):
	'''
	An easy to use function for plotting ellipses in Python 2.7!
		The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
		It can take a covariance matrix and plot contours from it.
		
		x_cent: float
			X coordinate center
		y_cent: float
			Y coordinate center
		theta_num: int
			Number of points to sample along ellipse from 0-2pi
		ax: matplotlib axis property
			A pre-created matplotlib axis
		plot_kwargs: dictionary
			matplotlib.plot() keyword arguments
		fill: bool
			A flag to fill the inside of the ellipse
		fill_kwargs: dictionary
			Keyword arguments for matplotlib.fill()
		cov: ndarray of shape (2,2)
			A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi
		mass_level: float
			if supplied cov, mass_level is the contour defining fractional probability mass enclosed
			for example: mass_level = 0.68 is the standard 68% mass
	'''
	clr = ['b','r','k']
	for ti,ellipsoid in enumerate(ellipsoids):
		x_cent = ellipsoid.get_center()[0] 
		y_cent = ellipsoid.get_center()[1]
		cov = ellipsoid.get_shape_mat()

		# Get Ellipse Properties from cov matrix
		if cov is not None:
			eig_vec,eig_val,u = np.linalg.svd(cov)
			# Make sure 0th eigenvector has positive x-coordinate
			if eig_vec[0][0] < 0:
				eig_vec[0] *= -1
			semimaj = np.sqrt(eig_val[0])
			semimin = np.sqrt(eig_val[1])
			if mass_level is None:
				multiplier = np.sqrt(2.279)
			else:
				distances = np.linspace(0,20,20001)
				chi2_cdf = chi2.cdf(distances,df=2)
				multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
			semimaj *= multiplier
			semimin *= multiplier
			phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
			if eig_vec[0][1] < 0 and phi > 0:
				phi *= -1
		# Generate data for ellipse structure
		theta = np.linspace(0,2*np.pi,theta_num)
		r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
		x = r*np.cos(theta)
		y = r*np.sin(theta)
		data = np.array([x,y])
		S = np.array([[semimaj,0],[0,semimin]])
		R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
		T = np.dot(R,S)
		data = np.dot(T,data)
		data[0] += x_cent
		data[1] += y_cent
		if ax is None:
			return_fig = True
			fig,ax = plt.subplots()
		if plot_kwargs is None:
			ax.plot(data[0],data[1],color=clr[ti],linestyle='-')
		else:
			ax.plot(data[0],data[1],**plot_kwargs)
		if fill == True:
			ax.fill(data[0],data[1],**fill_kwargs)

		ax.scatter(x_cent,y_cent,s=30)
		rx, ry = ellipsoid.get_eigenvals()
		ctr = ellipsoid.get_center()
		max_radius = max(0.1,np.max(abs(np.array([rx, ry])) + abs(ctr)))

	
	if halfspace is not None:
		hx = np.linspace(-max_radius,max_radius,5)
		hc = halfspace.get_normal()
		hy = (halfspace.get_rhs() - hc[0]*hx)*1.0/(hc[1]+1e-4)
		ax.plot(hx, hy, color='r',alpha=0.2)

		normaldata =np.zeros((2,2))
		normaldata[0,:] = ellipsoids[0].get_center()
		normaldata[1,:] = hc*1.0/(np.linalg.norm(hc,2)+1e-6)+ellipsoids[0].get_center()
		# print ctr
		# print hc
		ax.plot(normaldata[:,0],normaldata[:,1])

		hx = np.linspace(-max_radius*4,max_radius*4,50)
		hy = hx
		for i in hx:
			for j in hy:
				if halfspace.get_direction()=='leq':
					if (hc[0]*i + hc[1]*j <= halfspace.get_rhs()):
						hc2 = halfspace2.get_normal()
						if halfspace2 is not None and halfspace2.get_direction()=='leq':
							if (hc2[0]*i + hc2[1]*j <= halfspace2.get_rhs()):
								ax.scatter(i,j,color='g',marker='.',alpha=0.5,s=20)
						else:
							if (hc2[0]*i + hc2[1]*j >= halfspace2.get_rhs()):
								ax.scatter(i,j,color='g',marker='.',alpha=0.5,s=20)
				else:
					if (hc[0]*i + hc[1]*j >= halfspace.get_rhs()):
						hc2 = halfspace2.get_normal()
						if halfspace2 is not None and halfspace2.get_direction()=='leq':
							if (hc2[0]*i + hc2[1]*j <= halfspace2.get_rhs()):
								ax.scatter(i,j,color='g',marker='.',alpha=0.5,s=20)
						else:
							if (hc2[0]*i + hc2[1]*j >= halfspace2.get_rhs()):
								ax.scatter(i,j,color='g',marker='.',alpha=0.5,s=20)



	# #plot the degree-of-freedom-reduction hyperplane
	# x = np.linspace(-max_radius, max_radius,5)
	# #c = np.ones(2)
	# y = 1 - x
	# ax.plot(x,y,color='b', alpha=0.2)

	#plot the degree-of-freedom-reduction hyperplane (passing through center)
	ctr = ellipsoids[1].get_center()
	x = np.linspace(-max_radius, max_radius,5)
	#c = np.ones(2)
	y = (ctr[0] + ctr[1]) - x
	ax.plot(x,y,color='b', alpha=0.2)

	if custom_point is not None:
		ctr = np.around(custom_point,2)
		ax.scatter(ctr[0],ctr[1],color='b',marker='*',s=30) 
		ax.scatter(0,0,color='k',marker='o',s=30) 

	for axis in 'xy':
		getattr(ax, 'set_{}lim'.format(axis))((-3, 3))


	# x0, y0, dx, dy = ax.get_position().bounds
	# maxd = max(dx, dy)
	# width = 6 * maxd / dx
	# height = 6 * maxd / dy

	# plt.set_size_inches((width, height))

	plt.show()

##############################

if __name__=='__main__':
	np.random.seed(2018)
	np.set_printoptions(precision=4)



	ellipsoid = Ellipsoid(ctr=0.5*np.ones(2),shape_mat=np.array([[2,.3],[.1,1]]))
	hyperplane = Hyperplane(normal=np.ones(2)*1.0/np.sqrt(2),rhs=1.0/np.sqrt(2))
	plot_debug2D([ellipsoid,ellipsoid],halfspace=hyperplane)


	##debugging
	# ellipsoid = Ellipsoid(ctr=0.5*np.ones(3),shape_mat=np.array([[2,0,0],[0,1,0],[0,0,1]]))
	# hyperplane = Hyperplane(normal=np.ones(3)*1.0/np.sqrt(3),rhs=1.0/np.sqrt(3))
	# halfspace = hyperplane
	# plot_debug([ellipsoid],hyperplane=hyperplane,halfspace=halfspace,custom_point=np.random.rand(3))



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

	##debugging
	# dim = 4
	# ellipsoid = Ellipsoid(ctr=np.zeros(dim),shape_mat=np.eye(dim))
	# hyperplane = Hyperplane(normal=np.ones(dim)*1.0/np.sqrt(dim),rhs=1.0/np.sqrt(dim))
	# output_ellipsoid = get_ellipsoid_intersect_hyperplane(ellipsoid,hyperplane)
	# print output_ellipsoid.get_center()
