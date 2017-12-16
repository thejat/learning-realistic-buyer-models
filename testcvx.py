# Generate data for worst-case risk analysis.
import numpy as np
np.random.seed(2)
n = 5
mu = np.abs(np.random.randn(n, 1))/15
Sigma = np.random.uniform(-.15, .8, size=(n, n))
Sigma_nom = Sigma.T.dot(Sigma)
print "Sigma_nom ="
print np.round(Sigma_nom, decimals=2)

# Form and solve portfolio optimization problem.
# Here we minimize risk while requiring a 0.1 return.
from cvxpy import *
w = Variable(n)
ret = mu.T*w 
risk = quad_form(w, Sigma_nom)
prob = Problem(Minimize(risk), 
               [sum_entries(w) == 1, 
                ret >= 0.1,
                norm(w, 1) <= 2])
prob.solve()
print "w ="
print np.round(w.value, decimals=2)

# Form and solve worst-case risk analysis problem.
Sigma = Semidef(n)
Delta = Symmetric(n)
risk = quad_form(w.value, Sigma)
prob = Problem(Maximize(risk), 
               [Sigma == Sigma_nom + Delta, 
                diag(Delta) == 0, 
                abs(Delta) <= 0.2])
prob.solve()
print "standard deviation =", sqrt(quad_form(w.value, Sigma_nom)).value
print "worst-case standard deviation =", sqrt(risk).value
print "worst-case Delta ="
print np.round(Delta.value, decimals=2)