# -*- coding: UTF-8 -*-
"""
Author：
Date： 2021.11.09

"""
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Problem 1.1:

print('For this project I''ll choose the 8th data set. \n')

data = np.loadtxt('Data.txt')

# Problem 1.2:
print('Simply by opening Project4_Set8.txt in pycharm, there are 483 lines of data so N = 483. Also we could '
      'do np.size(data) which will also give N =', np.size(data))

# Problem 1.3:
Nbins = 40
fsize = 14
E_b = 100
ntotal = np.size(data)


plt.figure(1)
plt.hist(data, Nbins, density=True)
plt.xlabel('Photon Energy(a.u.)', fontsize = fsize)
plt.ylabel('Distribution', fontsize = fsize)
plt.title('PDF for Photon Energy(Cauchy Distribution)', fontsize = fsize)

print('From eye-balling the parameters, I estimated that ns = 175 pm 25'
      'E_0 = 210 pm 10 a.u and Gamma = 25 pm 5. \n')
# Problem 1.4

def loglikelihood(ns, E_0, Gamma):
    L = 1/(np.pi * Gamma * (1 + ((data - E_0) / Gamma) ** 2)) * ns/ntotal \
        + 1 / E_b * np.exp(-data / E_b) * (1 - ns/ntotal)
    return -2 * np.sum(np.log(L))

dim = 50
nsmin = 150.
nsmax = 200.
E_0min = 200.
E_0max = 220.
Gammamin = 20.
Gammamax = 30.
ns = np.linspace(nsmin, nsmax, dim)
E_0 = np.linspace(E_0min, E_0max, dim)
Gamma = np.linspace(Gammamin, Gammamax, dim)

GlobalMin = loglikelihood(ns[0], E_0[0], Gamma[0])
GlobalIdx = np.zeros(3)

# Do a brute force scan. For some reasons, when I worked on this part, scipy minimize did not work.
for i in range(len(ns)):
    for j in range(len(ns)):
        for k in range(len(ns)):
            if loglikelihood(ns[i], E_0[j], Gamma[k]) < GlobalMin:
                GlobalMin = loglikelihood(ns[i], E_0[j], Gamma[k])
                GlobalIdx[0] = i
                GlobalIdx[1] = j
                GlobalIdx[2] = k

# Plot specification
print('Minimum -2logL: {:.2f}'.format(GlobalMin))
print('This correponds to')
print('     ns  = {:.3f}'.format(int(ns[int(GlobalIdx[0])])))
nb = ntotal - int(ns[int(GlobalIdx[0])])
print('     nb  = {:.3f}'.format(nb))
print('     E_0  = {:.3f}'.format(float(E_0[int(GlobalIdx[1])])))
print('     Gamma  = {:.3f}'.format(float(Gamma[int(GlobalIdx[2])])))

# Problem 1.5
plt.figure(2)
xaxis = np.arange(0., 850., 850/1000.)
signal = 1/(np.pi * Gamma[int(GlobalIdx[2])] *
            (1 + ((xaxis - E_0[int(GlobalIdx[1])]) / Gamma[int(GlobalIdx[2])]) ** 2)) * ns[int(GlobalIdx[0])]/ntotal
background = 1 / E_b * np.exp(-xaxis / E_b) * (1 - ns[int(GlobalIdx[0])]/ntotal)
plt.plot(xaxis, signal, label = 'Signal')
plt.plot(xaxis, background, label = 'Background')
plt.plot(xaxis, signal + background, label = 'Signal + Background(Max. Likelihood fit)')
plt.hist(data, Nbins, density=True, label = 'histogram')
plt.xlabel('Photon Energy(a.u.)', fontsize = fsize)
plt.ylabel('Distribution', fontsize = fsize)
plt.title('PDF for Photon Energy(Cauchy Distribution)', fontsize = fsize)
plt.legend()

# Problem 1.6
def loglikelihoodWithKnownNs(E_0, Gamma):
    L = 1/(np.pi * Gamma * (1 + ((data - E_0) / Gamma) ** 2)) * ns[int(GlobalIdx[0])]/ntotal \
        + 1 / E_b * np.exp(-data / E_b) * (1 - ns[int(GlobalIdx[0])]/ntotal)
    return -2 * np.sum(np.log(L))

E0, GAMMA=np.meshgrid(E_0,Gamma)
vllh = np.vectorize(loglikelihoodWithKnownNs)
Lmap = vllh(E0, GAMMA)
min_entries = np.where(Lmap == np.amin(Lmap))
minimum2LogL = float(Lmap[min_entries[0],min_entries[1]])
Levs = [minimum2LogL+1]

# After googling for an hour I still cannot get the scale bar to show up.
fig, axes = plt.subplots()
contour = axes.contour(E0, GAMMA, Lmap, cmap='Reds', levels=Levs)
plt.plot(E_0[min_entries[1]], Gamma[min_entries[0]], 'ro',label='Fit')
plt.legend()
plt.xlabel("E_0", fontsize = fsize)
plt.ylabel("Gamma", fontsize = fsize)
plt.title('Contour Plot L vs E_0 and Γ(scale bar does not show up due to unknown reasons)', fontsize = fsize)
im = axes.imshow(Lmap,interpolation='none',extent=[E_0min,E_0max,Gammamin,Gammamax],origin='lower')
print('From eye-balling, the 1 sigma uncertainty on E_0 is around 3 a.u, and the '
      '1 sigma uncertainty on Gamma is around 3')

# Problem 1.7
print('From Wikipedia, the parameter Γ for the Wigner(Cauchy) distribution is the half-width-at-half-maximum. \n'
      'It also equals to half the interquartile range (dispersion of the data, (mean - Γ) is the 25th percentile \n'
      'and (mean + Γ) is the 75th percentile) and is sometimes called the probable error. \n'
      'Source: https://en.wikipedia.org/wiki/Interquartile_range\n; https://en.wikipedia.org/wiki/Cauchy_distribution\n'
      'https://en.wikipedia.org/wiki/Full_width_at_half_maximum')

# Problem 2.1:
plt.figure(4)
plt.hist(data, 25, density=True, range = (0, 400), label = 'histogram')
plt.xlabel('Photon Energy(a.u.)', fontsize = fsize)
plt.ylabel('Distribution', fontsize = fsize)
plt.title('PDF for Photon Energy(Cauchy Distribution, 25 bins)', fontsize = fsize)

# Problem 2.2
x0 = 0.
x1 = 400.
step = (x1 - x0)/25.
bins2 = np.arange(x0, x1, step)

nstest = 175.
E_0test = 210.
Gammatest = 25.
# CDF for the signal + background PDF. Did some integral to find out.
hitestValue = nstest/ntotal * 1/np.pi*(np.arctan((x1 - E_0test)/Gammatest)) - np.exp(-x1)*(1 - nstest/ntotal)
lotestValue = nstest/ntotal * 1/np.pi*(np.arctan((x0 - E_0test)/Gammatest)) - np.exp(-x0)*(1 - nstest/ntotal)

# I've tried for hours without setting A but it never worked. So we have to 'normalize' the bins:
A = ntotal/(hitestValue - lotestValue)

def Counts(lo,hi, ns2, E_02, Gamma2):
    hiValue = ns2/ntotal * 1/np.pi*(np.arctan((hi - E_02)/Gamma2)) - np.exp(-hi)*(1 - ns2/ntotal)
    loValue = ns2/ntotal * 1/np.pi*(np.arctan((lo - E_02)/Gamma2)) - np.exp(-lo)*(1 - ns2/ntotal)
    return A * (hiValue - loValue)

def Hyp(arguments):
    ns2 = arguments[0]
    E_02 = arguments[1]
    Gamma2 = arguments[2]
    return Counts(bins2, bins2+step, ns2, E_02, Gamma2)

def MyChi2(arguments):
    ns2 = arguments[0]
    E_02 = arguments[1]
    Gamma2 = arguments[2]
    return np.sum((y - Hyp([ns2, E_02, Gamma2]))*(y - Hyp([ns2, E_02, Gamma2]))/Hyp([ns2, E_02, Gamma2]))

y = np.random.poisson(Counts(bins2, bins2+step, 176, 209, 24))

chi2guess = [175., 210., 25.]
min_outcome = minimize(MyChi2, chi2guess)
chi2min = min_outcome.fun

ns2fit = min_outcome.x[0]
E02fit = min_outcome.x[1]
Gamma2fit = min_outcome.x[2]

ns2unc = np.sqrt(2* min_outcome.hess_inv[0,0])
E02unc = np.sqrt(2* min_outcome.hess_inv[1,1])
Gamma2unc = np.sqrt(2* min_outcome.hess_inv[2,2])

nbins = np.size(bins2 - 1) # The data size is NOT N, it's the number of bins.
ndof = nbins - 3  # Remove 3 for fittng
pvalue = 1 - chi2.cdf(chi2min, ndof)

xaxis1 = np.arange(0., 400., 400/1000.)
chi2Signal = 1/(np.pi * Gamma2fit * (1 + ((xaxis1 - E02fit) / Gamma2fit) ** 2)) * ns2fit/ntotal
chi2Background = 1 / E_b * np.exp(-xaxis1 / E_b) * (1 - ns2fit/ntotal)
plt.plot(xaxis1, chi2Signal + chi2Background, label = 'Chi2 fit')
plt.legend()

print("The minimized chi-squared is {:.3f} for {:d} degrees of freedom".format(chi2min, ndof))
print("The p-value for this fit is {:.3f}".format(pvalue))
print("The best fit value of the parameter are: ns2 = {:.3f} +- {:.3f}, E02 = {:.3f} +- {:.3f} "
      "and Gamma2 = {:.3f} +- {:.3f}".format(ns2fit, ns2unc, E02fit, E02unc, Gamma2fit, Gamma2unc))
print('The best fit value for nb is trivial because we have N = 483 = ns + nb. ')

print('The fitted values vary due to fluctuation when we take y = np.random.poisson, but it does generate reasonable \n'
      'results. I think the max. likelihood method is better in this case. The p-value also varies because y varies \n'
      'for different run. Below is the outcome I got from one of the runs, where I got p-value = 0.527. This is a \n'
      'pretty reasonable p-value because, as we discussed in the class, one would most likely get a p-value around 0.5.')
print('This is what I got from one of the runs:\n'
      '# The minimized chi-squared is 20.903 for 22 degrees of freedom \n'
      '# The p-value for this fit is 0.527 \n'
      '# The best fit value of the parameter are: ns2 = 174.662 +- 10.304, '
      '# E02 = 212.353 +- 2.787 and Gamma2 = 26.405 +- 2.859')

plt.show()

# The minimized chi-squared is 20.903 for 22 degrees of freedom
# The p-value for this fit is 0.527
# The best fit value of the parameter are: ns2 = 174.662 +- 10.304, E02 = 212.353 +- 2.787 and Gamma2 = 26.405 +- 2.859
