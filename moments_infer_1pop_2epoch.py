#!/usr/bin/env python
import sys,os
import numpy
from numpy import array,array_str
import moments

#fixed f=time/2Nb=T/nu
def two_epoch_nu_f(params, ns):
    """
    Instantaneous size change some time ago.
    params = (nu,T)
    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which size change happened (in units of 2*Na
       generations)
    ns: Number of samples in resulting Spectrum.
    """
    nu, f = params
    T = f * nu
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate([nu], T)
    return fs

#############################################################################

if len(sys.argv) < 2:
        print('prog AFS.in -iter(30) -BoundsFile  -ofile|stdout')
        print('example: ./moments_infer_1pop_2epoch.py  folded.AFS.1pop -Bbounds_1pop.txt -i50 -oout.1pop.2epo')
        sys.exit(1)

infile=sys.argv[1]
fout=''
modelx='two_epoch_nu_f'
nparm=2
for arg in sys.argv[2:]:
    if arg[:2] == '-i':
        iterlim= int(arg[2:])
    elif arg[:2] == '-B':
        Bstr = arg[2:]
        tar = arg[2:].split(',')
        BoundsFile= tar[0]
        if len(tar)==6:
            col_ini = int(tar[1])
            col_ptb = int(tar[2])
            col_upB = int(tar[3])
            col_lowB = int(tar[4])
            col_fix = int(tar[5])
        else:
            col_ini = 1
            col_ptb = 2
            col_upB = 3
            col_lowB = 4
            col_fix = 5
    elif arg[:2] == '-i':
        iterlim= int(arg[2:])
    elif arg[:2] == '-o':
        fout = arg[2:]


if not fout:
        fw =  sys.stdout
else:
        fw = open(fout, 'w')


data = moments.Spectrum.from_file(infile)
ns = data.sample_sizes

# get ini, bounds, fixed 
datB = numpy.genfromtxt(BoundsFile)
p_ini = datB[:,col_ini]
p_fold = datB[:,col_ptb]
p_upB = datB[:,col_upB]
p_lowB = datB[:,col_lowB]
p_fix = datB[:,col_fix]


if modelx == 'two_epoch_nu_f':
        func = two_epoch_nu_f
        nparm=2
        parm_names='nu,f'
else:
        print('#error:  model not knwon')
        sys.exit(1)


params = p_ini
upper_bound = list(p_upB)
lower_bound = list(p_lowB)
if numpy.all(numpy.isnan(p_fix)):  # nan to None in fixed
	fixedp= None
else:
	fixedp= [None]*nparm			
	for ii,pfix in enumerate(p_fix):
		if not numpy.isnan(pfix):
			fixedp[ii]  = pfix
p0 = moments.Misc.perturb_params(params, fold=p_fold, lower_bound=lower_bound, upper_bound=upper_bound)

popt = moments.Inference.optimize_log(p0, data, func,  
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound,
			           fixed_params=fixedp,
                                   verbose=10,  
                                   maxiter=iterlim)

model = func(popt, ns)
ll_opt = moments.Inference.ll_multinom(model, data)

fw.write('model:  %s (nu=Nb/Na   f=Tb/2Nb)\n' % modelx)
fw.write('Optimized_log-likelihood: %.3f\n' % ll_opt)
fw.write('model parameter names:   %s\n' % parm_names)
fw.write('model parameter values:  %s\n' % (numpy.array2string(popt,max_line_width=200,precision=3,separator=' ')[1:-1]))
fw.close()
