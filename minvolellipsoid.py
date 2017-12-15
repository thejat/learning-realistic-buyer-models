import numpy as np
import geometric
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import gamma
from mpl_toolkits.mplot3d import Axes3D

def get_min_vol_ellipsoid(ellipsoid,halfpsace):
	tolerance = 1e-4

	if ellipsoid.get_dimension() != halfpsace.get_dimension():
		print "Dimensional error!"
		return Ellipsoid()

	n = ellipsoid.get_dimension()
	A = ellipsoid.get_shape_mat()
	c = ellipsoid.get_center()
	p = halfpsace.get_pvec()

	b = np.dot(A,p)*1.0/np.sqrt(np.dot(p,np.dot(A,p)))
	#print "b=", b, "A=",A

	if halfpsace.get_direction()=='leq':
		new_c = c - b*1.0/(n+1)
	else:
		new_c = c + b*1.0/(n+1)

	new_A = ((n**2)*1.0/((n**2)-1))*(A - (2.0/(n+1))*np.outer(b,b))
	#print "new_A=", new_A, "new_c=", new_c

	return geometric.Ellipsoid(ctr=new_c,shape_mat=new_A)

def plot_debug2D(ellipsoids, halfspace=None, custom_point=None, halfspace2 = None, theta_num=1e3,ax=None,plot_kwargs=None, fill=False,fill_kwargs=None,mass_level=0.68):
	
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
	
	#plot hyperplane
	x = np.linspace(-max_radius, max_radius, 5)
	p=halfspace.get_pvec()
	c=halfspace.get_cvec()
	rhs=np.dot(p,c)
	y = ( rhs - p[0]*x )/p[1]
	plt.gca().set_aspect('equal', adjustable='box') # makes square plots
	ax.plot(x,y,color='b', alpha=0.2)

	ax.scatter(-0.9,0.5,s=30)
	plt.show()

if __name__=='__main__':
	ellipsoid = geometric.Ellipsoid(ctr=0.5*np.ones(2),shape_mat=np.array([[1,0],[0,1]]))
	print ellipsoid.get_membership(point=np.array([-0.9, 0.5]))
	halfspace = geometric.SpecialHalfspace(pvec= np.array([1,0]), cvec=ellipsoid.get_center(), direction= 'leq')
	ellipsoid2 = get_min_vol_ellipsoid(ellipsoid, halfspace)
	plot_debug2D([ellipsoid, ellipsoid2],halfspace=halfspace)
	