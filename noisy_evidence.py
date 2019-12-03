#%load_ext autoreload
#%autoreload 2
import numpy
from cosmoped_likelihood import CosMOPED
from DES import DES_like
import camb
from anesthetic import NestedSamples
import sys
import os


nested_dir = '/rds/user/wh260/hpc-work/pablo/CosmoChord/runs/chains/'
DES_dir = '/rds/user/wh260/hpc-work/pablo/CosmoChord/data/DES'

#nested_dir = '/data/will/data/pablo/runs_default/chains'
#DES_dir = '/home/will/Projects/CosmoChord/data/DES'

DES_samples = NestedSamples(root=os.path.join(nested_dir,'DES'))
planck_samples = NestedSamples(root=os.path.join(nested_dir,'planck'))
DES_planck_samples = NestedSamples(root=os.path.join(nested_dir,'DES_planck'))

# create a CosMOPED object
TT2015 = CosMOPED('compression_vectors/output/LambdaCDM/', year=2015, spectra='TT', use_low_ell_bins=True)

# create a DES object
DESY1 = DES_like(os.path.join(DES_dir,'DES_1YR_final.dataset')) 


def random_posterior_sample(samples):
    i = numpy.random.choice(len(samples),p=samples.weight/samples.weight.sum())
    return samples.iloc[i]

def get_des_params(H0, ombh2, omch2, tau, As, ns):
    pars = camb.CAMBparams()
    pars.set_for_lmax(2510)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, YHe=0.245341, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.WantTransfer=True
    results = camb.get_results(pars)
    return numpy.concatenate([results.get_sigma8(), [ombh2+omch2/(H0/100)]])

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
DES_samples['1e-9A'] = 1e-9*DES_samples['A']
params = ['H0', 'omegabh2', 'omegach2', 'tau', '1e-9A', 'ns']
planck_cov = planck_samples[params].cov()
planck_mu = planck_samples[params].mean()
planck_L = numpy.linalg.cholesky(planck_cov)
planck_Linv = numpy.linalg.inv(planck_L)

DES_cov = DES_samples[params].cov()
DES_mu = DES_samples[params].mean()
DES_L = numpy.linalg.cholesky(DES_cov)
DES_Linv = numpy.linalg.inv(DES_L)

from scipy.optimize import minimize
#from camb import CAMBError

if sys.argv[1] == '0':
    theta_planck = theta_DES = None
    planck = TT2015
    DES = DESY1
else:
    numpy.random.seed(int(sys.argv[1]))
    if sys.argv[2] == 'planck':
        theta_planck = theta_DES = random_posterior_sample(planck_samples)
    elif sys.argv[2] == 'DES': 
        theta_planck = theta_DES = random_posterior_sample(DES_samples)
    elif sys.argv[2] == '2cosmo':  
        theta_planck = random_posterior_sample(planck_samples)
        theta_DES = random_posterior_sample(DES_samples)

    planck = planck_realization(*theta_planck[params])
    DES = DES_realization(*theta_DES[params]) 


def f(x, DES, planck):
    if planck is None:
        t = DES_mu + DES_L @ x
    else:
        t = planck_mu + planck_L @ x
    try:
        l = loglike(*t, DES=DES, planck=planck) 
        print(x,l)
        return -l
    except:
        return 1e30

filename =  'data/' + sys.argv[2] + '_' + sys.argv[1] + '.txt'

sol = minimize(f,numpy.random.randn(6), args=(DES,planck),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
with open(filename, "a") as myfile:
    theta = numpy.array(planck_mu + planck_L @ sol.x)
    print('DES+planck theta:', numpy.concatenate([theta,get_des_params(*theta)]), file=myfile)
    print('DES+planck logL:', -sol.fun, file=myfile)

print('--------------------------------------------')
sol = minimize(f,numpy.random.randn(6), args=(None,planck),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
with open(filename, "a") as myfile:
    theta = numpy.array(planck_mu + planck_L @ sol.x)
    print('planck theta:', numpy.concatenate([theta,get_des_params(*theta)]), file=myfile)
    print('planck logL:', -sol.fun, file=myfile)

print('--------------------------------------------')
sol = minimize(f,numpy.random.randn(6), args=(DES,None),method='Nelder-Mead',options={'initial_simplex':numpy.random.randn(7,6)})
with open(filename, "a") as myfile:
    theta = numpy.array(DES_mu + DES_L @ sol.x)
    print('DES theta:', numpy.concatenate([theta,get_des_params(*theta)]), file=myfile)
    print('DES logL:', -sol.fun, file=myfile)
