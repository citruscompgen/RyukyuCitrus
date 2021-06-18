#!/usr/bin/env python
import sys,gzip,os,os.path,time
import numpy
from numpy import array,array_str
import numpy.ma as ma
import moments


#based on moments_infer_1pop.py
#2021  option to produce AFS from model by using narrow (say 1%) variation of parm values in bound.txt
#eg_folded n=16
"""
17 folded
22848149 40759 21178 14772 14459 10710 10484 9610 6323 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
"""

#fixed f=time/2N=T/nu
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
        print('prog AFS.in -iterlim(10) -BoundsFile[,col_ini,col_perturb,col_up,col_low,col_fix] [-model,EXP,nparm] -ofile|stdout [-seqLen,mug]')
        print('-B: col_perturb:  will perturb this parm up or down by 2**val')
        print('-seqLen,mug  tot_elg_sites,mug for infering N_anc and abs. time and pop size eg 2.73e8,7e-9')
        print('-Bbounds: eg parm1 fixed at value=p1:  parm1 p1 1 p1 p1 p1')
        print('-Bbounds: eg parm2 variable:           parm2 1 1 10 1e-3 none')
        print('rmk:  moments assume theta_anc=1')
        print('NEW: gen. AFS from model by using narrow (say 1%) variation of 1 parm and fix the others in bound.txt')
        print('eg2: all.RK.fold -i50 -Bbounds_nu0.2,3,4,2,1,5  -mtwo_epoch_nu_f,0,2 -omomo.2epoFxNu0.2 -llog.i50 -s2.298e7,1e-8')
        sys.exit(1)

infile=sys.argv[1]
fout=''
modelx='two_epoch_nu_f'
EXP=0
nparm=2
seqlen=1
mug=5e-9
for arg in sys.argv[2:]:
    if arg[:2] == '-i':
        iterlim= int(arg[2:])
    elif arg[:2] == '-m':
        tar= arg[2:].split(',')
        modelx=tar[0]
        EXP=int(tar[1])
        nparm=int(tar[2])
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
    elif arg[:2] == '-s':
        seqlen, mug = list(map(lambda x: float(x), arg[2:].split(',')))

#print('#model=%s EXP=%d nparm=%d' % (modelx,EXP,nparm))

if not fout:
        fw =  sys.stdout
else:
        fw = open(fout, 'w')


time0 = time.time()
data = moments.Spectrum.from_file(infile)
ns = data.sample_sizes

# get ini, bounds, fixed 
datB = numpy.genfromtxt(BoundsFile)
p_ini = datB[:,col_ini]
p_fold = datB[:,col_ptb]
p_upB = datB[:,col_upB]
p_lowB = datB[:,col_lowB]
p_fix = datB[:,col_fix]
if nparm != datB.shape[0]:
	print('#error: nparm does not match BoundsFile')
	sys.exit(1)
ncolB = datB.shape[1]

if nparm % 2 ==0:
	np_2=int(nparm/2)

parm_names=''
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
theta_optm = moments.Inference.optimal_sfs_scaling(model, data)  # data.sum()/model.sum() for common non-masked
ll_opt = moments.Inference.ll_multinom(model, data)
N_anc = theta_optm/(4*mug*seqlen) 
theta_anc = theta_optm/seqlen 


#print('ncolBound=%d' % ncolB)
pmar = numpy.zeros((nparm,ncolB+4))
pmar[:,0] = datB[:,col_ini]
pmar[:,1] = datB[:,col_ptb]
pmar[:,2] = datB[:,col_upB]
pmar[:,3] = datB[:,col_lowB]
pmar[:,4] = datB[:,col_fix]
pmar[:,5] = p0
pmar[:,6] = popt
pmar[:,7] = (datB[:,col_upB] - popt)/datB[:,col_upB] < 0.02  # 2% cutoff
for i in list(range(nparm)):
        if (datB[i,col_lowB] == 0 and popt[i] < 0.02) or ( (popt[i] - datB[i,col_lowB])/datB[i,col_lowB] < 0.02):
                pmar[i,9] = 1

fw.write('Optimized theta and log-likelihood: %s\n' % repr([theta_optm,ll_opt])[1:-1])
fw.write('Opt_ll/prm: %.3f  %s\n' % (ll_opt, numpy.array2string(popt,max_line_width=200,precision=3,separator=' ')[1:-1]))
fw.write('model/parm_names: %s  %s\n' % (modelx,parm_names))
fw.write('parameters:   ini0    perturb_fold   upperB    lowerB     fixed   ini_perturbed   Optim  nearUpB(2%)   nearLowB(2%):\n')
fw.write('%s\n' % numpy.array2string(pmar,max_line_width=200,precision=3,separator=' ').replace('[','').replace(']',''))

model = moments.Inference.optimally_scaled_sfs(model, data)   # rescaled unfolded model by theta
full_model=numpy.vstack((numpy.arange(1,ns[0]),model.compressed())).transpose()
fw.write('\n\n#1 model-full-AFS:\n%s\n' % numpy.array2string(full_model,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']',''))

if data.folded and not model.folded:
	model = model.fold()
	folded_model=numpy.vstack((numpy.arange(1,int(ns[0]/2+1)),model.compressed())).transpose()
	fw.write('\n\n#2 model-folded/unmasked-AFS:\n%s\n' % numpy.array2string(folded_model,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']',''))
masked_model, masked_data = moments.Numerics.intersect_masks(model, data)  # ## mask model as in data
nmask=masked_model.mask.sum()
n_unmsk = masked_model.count()

S_data = data.S()
thetaW_data_knt = data.Watterson_theta()
thetaW_data = data.Watterson_theta()/seqlen
pi_data = data.pi()/seqlen
Tajima_D_data=data.Tajima_D()
S_model = masked_model.S()
thetaW_model = masked_model.Watterson_theta()/seqlen
pi_model = masked_model.pi()/seqlen
Tajima_D_model=masked_model.Tajima_D()

fw.write('\n\n#3 data[S=seg.site; thetaW=S/[slen*sum(1/i)]; pi=sum(wi*i*(n-i))\n')
fw.write('S_data= %.0f   S_model= %.0f\n' % (S_data,S_model))
fw.write('thetaW_data= %.2e  thetaW_model= %.2e\n' % (thetaW_data,thetaW_model))
fw.write('pi_data= %.2e   pi_model= %.2e\n' % (pi_data,pi_model))
fw.write('TajimaD_data= %.2e   TajimaD_model= %.2e\n' % (Tajima_D_data,Tajima_D_model))
fw.write('theta_anc= %.3e\n' % theta_anc)


###### pop AFS
DEV = (masked_model - masked_data) /masked_model
idx_af1=ma.array(data=numpy.arange(0,ns[0]+1),mask=ma.getmaskarray(masked_data))
dat_const=thetaW_data_knt/numpy.arange(0,ns[0]+1)
if data.folded:
    dat_const=thetaW_data_knt/numpy.arange(0,ns[0]+1) + thetaW_data_knt/numpy.arange(ns[0],-1,-1)
    if ns[0] % 2 ==0:
        dat_const[int(ns[0]/2)] *= 0.5
AFS_const=ma.array(data=dat_const,mask=ma.getmaskarray(masked_data))
af1d = ma.vstack((idx_af1.compressed(),masked_data.compressed(),masked_model.compressed(),DEV.compressed())).transpose()
af1d = array(af1d)
fw.write('\n\n#4 AFS  data  model  dev=(model-data)/model:\n%s\n' % numpy.array2string(af1d,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']',''))
srti = af1d[:,3].argsort()
fw.write('#dev=(M-D)/M: max[AF=%.0f]= %.4f   min[AF=%.0f]= %.4f\n' % (af1d[srti[-1],0],af1d[srti[-1],3],af1d[srti[0],0],af1d[srti[0],3]))

###data model const
af1e = ma.vstack((idx_af1.compressed(),masked_data.compressed(),masked_model.compressed(),AFS_const.compressed(),DEV.compressed())).transpose()
af1e = array(af1e)
fw.write('\n\n#5 AFS  data  model  constPop  dev=(model-data)/model:\n%s\n' % numpy.array2string(af1e,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']',''))

### output model parameters
fw.write('\n\n#6 optm parameters w/ seqlen=%.3e and mug=%.3e\n' % (seqlen,mug))
fw.write('thetaW_data= %.2e  thetaW_model= %.2e\n' % (thetaW_data,thetaW_model))
fw.write('pi_data= %.2e   pi_model= %.2e\n' % (pi_data,pi_model))
fw.write('theta_anc= %.3e\n' % theta_anc)
fw.write('N_anc[theta/4*mug*seqlen]= %.3e\n' % N_anc)
if modelx=='two_epoch_nu_f':
    N2 = popt[0]*N_anc
    f=popt[1] # f=T/nu=time/2N
    T = f*2*N2
    fw.write('\n# model=%s parm=nuB,f[=time/2Nb]\n' % modelx)
    fw.write('f= %.3e\n' % f)
    fw.write('N2= %.3e\n' % N2)
    fw.write('T2/gen= %.3e\n' % T)
    theta_0 = 4*mug*N2
    f_0 = 1 - numpy.exp(-f)
    pi_theory = theta_0 * f_0 + theta_anc * (1- f_0)
    fw.write('inbr_coef=f_0= %.3e\n' % f_0)
    fw.write('theta_0= %.3e\n' % theta_0)
    fw.write('pi_theory= %.3e\n' % pi_theory)
    popT=array([popt[0],popt[1]*popt[0]])  # [nu,T] 

fw.write('#time passed %.1f sec\n' % ( time.time() - time0))
fw.close()
sys.exit(1)
