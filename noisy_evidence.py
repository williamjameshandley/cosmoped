#%load_ext autoreload
#%autoreload 2
import numpy
from cosmoped_likelihood import CosMOPED
from DES import DES_like
import camb
from anesthetic import NestedSamples
import sys

DES_samples = NestedSamples(root='/data/will/data/pablo/runs_default/chains/DES')
planck_samples = NestedSamples(root='/data/will/data/pablo/runs_default/chains/planck')
DES_planck_samples = NestedSamples(root='/data/will/data/pablo/runs_default/chains/DES_planck')

# create a CosMOPED object
TT2015 = CosMOPED('compression_vectors/output/LambdaCDM/', year=2015, spectra='TT', use_low_ell_bins=True)

# create a DES object
DESY1 = DES_like('/home/will/Projects/CosmoMC/data/DES/DES_1YR_final.dataset') 


def random_posterior_sample(samples):
    i = numpy.random.choice(len(samples),p=samples.weight/samples.weight.sum())
    return samples.iloc[i]

def planck_realization(H0, ombh2, omch2, tau, As, ns):
    pars = camb.CAMBparams()
    pars.set_for_lmax(2510)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, YHe=0.245341, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    ellmin=2
    Dltt = powers['total'][ellmin:,0]
    Dlee = powers['total'][ellmin:,1]
    Dlte = powers['total'][ellmin:,3]
    return TT2015.ppd_resample(Dltt, Dlee, Dlte, ellmin)

def DES_realization(H0, ombh2, omch2, tau, As, ns):
    pars = camb.CAMBparams()
    pars.set_for_lmax(2510)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, YHe=0.245341, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    results, PKdelta, PKWeyl = DESY1.get_camb_theory(pars)
    theory = DESY1.get_theory(pars, results, PKdelta, PKWeyl)
    return DESY1.ppd_resample(theory)

def loglike(H0, ombh2, omch2, tau, As, ns, DES=None, planck=None):
    logl = 0
    ellmin = 2
    pars = camb.CAMBparams()
    pars.set_for_lmax(2510)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, YHe=0.245341, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)

    if DES is not None:
        results, PKdelta, PKWeyl = DES.get_camb_theory(pars)
        theory = DES.get_theory(pars, results, PKdelta, PKWeyl)
        logl += -0.5*DES.chi_squared(theory) 
    else:
        results = camb.get_results(pars)

    if planck is not None:
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        Dltt = powers['total'][ellmin:,0]
        Dlee = powers['total'][ellmin:,1]
        Dlte = powers['total'][ellmin:,3]
        logl += planck.loglike(Dltt, Dlte, Dlee, ellmin)

    return logl



# Sample from planck posterior
planck_samples['1e-9A'] = 1e-9*planck_samples['A']
params = ['H0', 'omegabh2', 'omegach2', 'tau', '1e-9A', 'ns']
cov = planck_samples[params].cov()
mu = planck_samples[params].mean()
L = numpy.linalg.cholesky(cov)
Linv = numpy.linalg.inv(L)

from scipy.optimize import minimize
from camb import CAMBError

if sys.argv[1] == '0':
    theta = None
    planck = TT2015
    DES = DESY1
else:
    numpy.random.seed(int(sys.argv[1]))
    if sys.argv[2] == 'planck':
        theta = random_posterior_sample(planck_samples)
    elif sys.argv[2] == 'DES': 
        theta = random_posterior_sample(DES_samples)
    planck = planck_realization(*theta[params])
    DES = DES_realization(*theta[params]) 

def f(x, DES, planck):
    t = mu + L @ x
    try:
        l = loglike(*t, DES=DES, planck=planck) 
        print(x,l)
        return -l
    except CAMBError:
        return 1e30

print('--------------------------------------------')
print('true theta:', theta)
sol = minimize(f,numpy.random.randn(6), args=(DES,planck),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
print('DES+planck theta:' mu + L @ sol.x)
print('DES+planck logL:' -sol.fun)

sol = minimize(f,numpy.random.randn(6), args=(None,planck),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
print('planck theta:' mu + L @ sol.x)
print('planck logL:' -sol.fun)

sol = minimize(f,numpy.random.randn(6), args=(DES,None),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
print('DES theta:' mu + L @ sol.x)
print('DES logL:' -sol.fun)
