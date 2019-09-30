#%load_ext autoreload
#%autoreload 2
import numpy as np
from cosmoped_likelihood import CosMOPED

ls, Dltt, Dlte, Dlee = np.genfromtxt('Dl_planck2015fit.dat', unpack=True)
ellmin=int(ls[0])

path='compression_vectors/output/LambdaCDM/'

# create a CosMOPED object
TT2015=CosMOPED(path, year=2015, spectra='TT', use_low_ell_bins=False)

# Need to write code that links theta to Dltt etc
# ...................

# Generate a new sample
sample = TT2015.ppd_resample(Dltt, Dlte, Dlee, ellmin)

sample.loglike(Dltt, Dlte, Dlee, ellmin)
TT2015.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default



from DES import DES_like
DES = DES_like('/home/will/Projects/CosmoMC/data/DES/DES_1YR_final.dataset')

pars = DES.get_test_pars()
results, PKdelta, PKWeyl = DES.get_camb_theory(pars)
theory = DES.get_theory(pars, results, PKdelta, PKWeyl)
sample = DES.ppd_resample(theory)
sample.chi_squared(theory)
DES.chi_squared(theory)
numpy.linalg.inv(DES.covinv)



