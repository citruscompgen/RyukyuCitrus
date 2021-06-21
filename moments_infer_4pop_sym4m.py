#!/usr/bin/env python
import sys,os
import numpy
from numpy import array,array_str
import moments

# for 4-population tree: (pop4,(pop3,(pop2,pop1)))
# nui = Ni/Na Na=N_ancestral
# Ti: time duration for i pops in units of 2*Na
# Mij: migrations
def pop4_MPRS_sym4m(params, ns): # nu3b nu3
    nu1B,nu1C,nu1F,nu2,nu3,nu4, Mb2,Mc3,M13,M14, T2,T3,T4 = params
    n1,n2,n3,n4 = ns
    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3+n4)
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, n1+n3+n4, n2)
    mig1 = numpy.array([[0, Mb2],[Mb2, 0]])
    fs.integrate([nu1B, nu2], T2, m=mig1) # default dt_fac=0.02 m=none
    fs = moments.Manips.split_2D_to_3D_1(fs, n1+n4, n3)
    mig2 = numpy.array([[0, 0, Mc3], [0, 0, 0], [Mc3, 0, 0]])
    fs.integrate([nu1C,nu2,nu3], T3, m=mig2)
    fs = moments.Manips.split_3D_to_4D_1(fs, n1, n4)
    mig3 = numpy.array([[0,0,M13,M14], [0,0,0,0],[M13,0,0,0], [M14,0,0,0]])
    fs.integrate([nu1F,nu2,nu3,nu4], T4, m=mig3)
    return fs

#############################################################################

if len(sys.argv) < 2:
    print('prog in.sfs -iter(50) -Bounds -ofile') 
    #print('-models: MPRS_sym4m= pop4_MPRS_sym4m((nu1B,nu1C,nu1F,nu2,nu3,nu4, Mb2,Mc3,M13,M14, T2,T3,T4),..)  10+4prm')
    print('example: ./moments_infer_4pop_sym4m.py MPRS_4pop.folded -i40 -Bbounds_4pop.txt  -oout.4pop') 
    sys.exit(1)

npop=4
infile=sys.argv[1]
fout=''
modelx='MPRS_sym4m'
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
        elif arg[:2] == '-o':
                fout = arg[2:]

if not fout:
        fw =  sys.stdout
else:
        fw = open(fout, 'w')

data = moments.Spectrum.from_file(infile)  # 
ns = data.sample_sizes   # ns is array not list eg array([2, 2, 2])

# get ini, bounds, fixed 
datB = numpy.genfromtxt(BoundsFile)
p_ini = datB[:,col_ini]  # array not list
p_fold = datB[:,col_ptb]
p_upB = datB[:,col_upB]
p_lowB = datB[:,col_lowB]
p_fix = datB[:,col_fix]
nparm = datB.shape[0]

if modelx == 'MPRS_sym4m':   # submodel of ABe_AeCe_4m w/ riF=1 i =1,2,3
	func = pop4_MPRS_sym4m
	parm_names='nu1B,nu1C,nu1F,nu2,nu3,nu4, Mb2,Mc3,M13,M14, T2,T3,T4' # 10+4 parm
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
			if pfix < p_lowB[ii] or pfix > p_upB[ii]:
				print('#erro: fixed value out of bound: lowB= %.3e  upB= %.3e fix= %.3e' % (p_lowB[ii],p_upB[ii],pfix))
				sys.exit(1)

fw.write('fs_infile= %s  sample_sz=%s  max_iter= %d   Model=%s   BoundsInfo=%s\n' % (infile,ns,iterlim,modelx,Bstr))

if iterlim > 0:
    p0 = moments.Misc.perturb_params(params, fold=p_fold, lower_bound=lower_bound, upper_bound=upper_bound)
    popt = moments.Inference.optimize_log(p0, data, func, lower_bound=lower_bound, upper_bound=upper_bound, fixed_params=fixedp, verbose=len(p0), maxiter=iterlim)
else:
    p0 = p_ini
    popt = p0

model = func(popt, ns)
ll_opt = moments.Inference.ll_multinom(model, data)

fw.write('Optimized log-likelihood:  %.1f\n' % ll_opt) 
fw.write('paarmeter names:   %s\n' % parm_names)
fw.write('parameter values:  %s\n' % numpy.array2string(popt,max_line_width=200,precision=3,separator=' ')[1:-1])
fw.close()
