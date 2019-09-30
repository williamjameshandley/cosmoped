#%load_ext autoreload
#%autoreload 2
import numpy as np
from cosmoped_likelihood import CosMOPED

ls, Dltt, Dlte, Dlee = np.genfromtxt('Dl_planck2015fit.dat', unpack=True)
ellmin=int(ls[0])

path='compression_vectors/output/LambdaCDM/'

# create a CosMOPED object
TT2015=CosMOPED(path, year=2015, spectra='TT', use_low_ell_bins=False)

# Generate a new sample
sample = TT2015.ppd_resample(Dltt, Dlte, Dlee, ellmin)

sample.loglike(Dltt, Dlte, Dlee, ellmin)
TT2015.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
