#!/usr/bin/env python
import sys,gzip,os,os.path,time
import numpy
from numpy import array,array_str
import numpy.ma as ma
import moments

##OK_py3 
##based on moments_infer_4pop_v0.2.py
##########################################################################################

# MA=1, PU=2,  RK=3 split from MA, MS=4 split from MA topo=(((1,4),3),2)
# 10 parm:  T[2,3,4-lineage] = T2,3,4
#       pop sz history: Na=>N1b=>N1c=>N1f  N2, N3, N4
## 4 mig rates b/t split: T2:Mb2  T3:Mc3 T4:M14,M13
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


# AFS is numpy array (could be masked)
def SS4_2pop(AFS):
	n_s = AFS.sample_sizes
	S_fixed = AFS[[0,n_s[0]],[n_s[1],0]].sum()
	S_share = AFS[1:n_s[0],1:n_s[1]].sum()
	S_excl1 = AFS[1:n_s[0],[0,n_s[1]]].sum()
	S_excl2 = AFS[[0,n_s[0]],1:n_s[1]].sum()
	SS4 = array([S_fixed,S_share,S_excl1,S_excl2])
	return SS4

#############################################################################

if len(sys.argv) < 2:
    print('prog in.sfs -iterlim(30) -Bounds[,col_ini,col_perturb,col_up,col_low,col_fix] [-model"MPRS_sym4m"] -o[dir,,]ofile [-seqLen,mug]')
    print('-models: MPRS_sym4m= pop4_MPRS_sym4m((nu1B,nu1C,nu1F,nu2,nu3,nu4, Mb2,Mc3,M13,M14, T2,T3,T4),..)  10+4prm')
    print('-B: col_perturb:  will perturb this parm up or down by 2**val')
    print('-B: boundsFile row names not used but are ordered')
    print('-seqLen,mug  tot_elg_sites,mug for infering abs. time and pop size eg 2.73e8,7e-9')
    print("example: moments_infer_4pop_pub.py MPRS.fold -i30 -Bbounds.txt,3,4,2,1,5 -m'MPRS_sym4m' -out.4pop")
    sys.exit(1)

npop=4
prn_4D_AFS=0
infile=sys.argv[1]
fout=''
modelx='MPRS_sym4m'
seqlen=1
mug=1e-8
pts_l=[]
for arg in sys.argv[2:]:
        if arg[:2] == '-i':
                iterlim= int(arg[2:])
        elif arg[:2] == '-m':
                modelx= arg[2:]
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
                tar = arg[2:].split(',,')
                if len(tar) ==2:
                    odir,fn = arg[2:].split(',,')
                    fout = odir + '/' + fn
                else:
                    odir='.'
                    fn=tar[0]
                    fout = fn
        elif arg[:2] == '-g':
                pts_l = list(map(lambda x: int(x), arg[2:].split(',')))
                if len(pts_l) != 3:
                    print('error: give 3 grid sizes')
                    sys.exit(1)
        elif arg[:2] == '-s':
                seqlen, mug = list(map(lambda x: float(x), arg[2:].split(',')))

if odir !=  ".":
    os.system('mkdir -p %s' % odir)
if not fn:
        fw =  sys.stdout
else:
        fw = open(fout, 'w')


time0 = time.time()
data = moments.Spectrum.from_file(infile)  # 
ns = data.sample_sizes   # ns is array not list eg array([2, 2, 2])
ns_lst =list(ns)
print(ns_lst)
ns_mx = ns.max()
nmask_data=data.mask.sum()   # 11 for 3-dip folded
n_unmsk_data = data.count()  # 16 for 3-dip folded
tot_sz= nmask_data + n_unmsk_data
dat_max = data.max()
dat_min = data.min()
dat_count_10fold = (data >= dat_max/10.).sum()  # num entries >= max/10
dat_count_20fold = (data >= dat_max/20.).sum()
dat_count_100fold = (data >= dat_max/100.).sum()

# get ini, bounds, fixed 
datB = numpy.genfromtxt(BoundsFile)
p_ini = datB[:,col_ini]  # array not list
p_fold = datB[:,col_ptb]
p_upB = datB[:,col_upB]
p_lowB = datB[:,col_lowB]
p_fix = datB[:,col_fix]
nparm = datB.shape[0]
ncolB = datB.shape[1]

parm_names=''
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
Fst_data= data.Fst()
Fst2d_D=numpy.zeros([npop,npop])
Fst2d_M=numpy.zeros([npop,npop])
for i in range(npop-1):
    for j in range(i+1,npop):
        margi= list(set(range(npop))-set([i,j]))
        Fst2d_D[i,j]=data.marginalize(margi).Fst()
        Fst2d_D[j,i] = Fst2d_D[i,j]

# optimize_log_lbfgsb() : 
if iterlim > 0:
    p0 = moments.Misc.perturb_params(params, fold=p_fold, lower_bound=lower_bound, upper_bound=upper_bound)
    popt = moments.Inference.optimize_log(p0, data, func, lower_bound=lower_bound, upper_bound=upper_bound, fixed_params=fixedp, verbose=len(p0), maxiter=iterlim)
else:
    p0 = p_ini
    popt = p0

model = func(popt, ns)
theta_optm = moments.Inference.optimal_sfs_scaling(model, data)  # data.sum()/model.sum() for common non-masked
ll_opt = moments.Inference.ll_multinom(model, data)
###anc pop size  vs theta_optm
N_anc = theta_optm/(4*mug*seqlen)


pmar = numpy.zeros((nparm,10))
pmar[:,0] = datB[:,col_ini]
pmar[:,1] = datB[:,col_ptb]
pmar[:,2] = datB[:,col_upB]
pmar[:,3] = datB[:,col_lowB]
pmar[:,4] = datB[:,col_fix]
pmar[:,5] = p0
pmar[:,6] = popt
pmar[:,7] = (datB[:,col_upB] - popt)/datB[:,col_upB] < 0.02  # 2% cutoff
for i in range(nparm):
        pmar[i,9]=i+1  # parm idx
        if (datB[i,col_lowB] == 0 and popt[i] < 0.02) or ( datB[i,col_lowB] !=0 and (popt[i] - datB[i,col_lowB])/datB[i,col_lowB] < 0.02):
                pmar[i,8] = 1

fw.write('Optimized theta (scaling factor) and log-likelihood: %s\n' % repr([theta_optm,ll_opt]))
fw.write('Opt_ll/prm: %.1f  %s\n' % (ll_opt, numpy.array2string(popt,max_line_width=200,precision=3,separator=' ')[1:-1]))
fw.write('parm_names: %s\n' % parm_names)
fw.write('parameters:   ini0    perturb_fold   upperB    lowerB     fixed   ini_perturbed   Optim  nearUpB(2%)   nearLowB(2%)  idx:\n')
fw.write('%s\n' % numpy.array2string(pmar,max_line_width=200,precision=3,separator=' '))

model = moments.Inference.optimally_scaled_sfs(model, data)   # rescaled unfolded model by theta
if data.folded and not model.folded:
        model = model.fold()
masked_model, masked_data = moments.Numerics.intersect_masks(model, data)  # ## mask model as in data
nmask=masked_model.mask.sum()
n_unmsk = masked_model.count()
fw.write('\n\nn_unmask_data/model=  %d  %d  nmask_data/model= %d  %d\n' % (n_unmsk_data,n_unmsk,nmask_data,nmask))
dev_jAFS = (masked_model - masked_data)/masked_data
if prn_4D_AFS:
    fw.write('\n\nOptimized model_Rescaled/Folded/Masked:\n%s\n' % numpy.array2string(masked_model,max_line_width=500,precision=3,separator=' '))
    fw.write('\n\ndata_jAFS/Folded/Masked:\n%s\n' % numpy.array2string(masked_data,max_line_width=500,precision=3,separator=' '))
    fw.write('\n\ndev=(model-data)/data:\n%s\n' % numpy.array2string(dev_jAFS,max_line_width=500,precision=3,separator=' '))

####
if modelx.startswith('MPRS_'):  # nu1A,nu1B,nu1C,nu2A,nu2B,nu3B,nu3C, [M1b, M12, M13, M23],z1,z2,z3,T2, T3
	N_1b=popt[0]*N_anc
	N_1c=popt[1]*N_anc
	N_1F=popt[2]*N_anc
	N_2=popt[3]*N_anc
	N_3=popt[4]*N_anc
	N_4=popt[5]*N_anc
	T2 =popt[-3]*2*N_anc
	T3 =popt[-2]*2*N_anc
	T4 =popt[-1]*2*N_anc
	F_1b =1- numpy.exp(-T2/(2*N_1b))
	F_1c =1- numpy.exp(-T3/(2*N_1c))
	F_1F =1- numpy.exp(-T4/(2*N_1F))
	F_4 =1- numpy.exp(-T4/(2*N_4))
	F_3 =1- numpy.exp(-(T3+T4)/(2*N_3))
	F_2 =1- numpy.exp(-(T2+T3+T4)/(2*N_2))
	thet_a = 4*mug*N_anc  # per bp
	thet0_1b = 4*mug*N_1b  
	thet0_1c = 4*mug*N_1c  
	thet0_1F = 4*mug*N_1F
	thet0_2 = 4*mug*N_2
	thet0_3 = 4*mug*N_3
	thet0_4 = 4*mug*N_4
	theta_1 = thet0_1F * F_1F + thet0_1c * (1-F_1F)*F_1c + thet0_1b * (1-F_1F)*(1-F_1c)*F_1b + thet_a * (1-F_1F)*(1-F_1c)*(1-F_1b) 
	theta_4 = thet0_4 * F_4 + thet0_1c * (1-F_4)*F_1c + thet0_1b * (1-F_4)*(1-F_1c)*F_1b + thet_a * (1-F_4)*(1-F_1c)*(1-F_1b) 
	theta_3 = thet0_3 * F_3 + thet0_1b * (1-F_3)*F_1b + thet_a * (1-F_3)*(1-F_1b) 
	theta_2 = thet0_2 * F_2 + thet_a * (1-F_2)
	fw.write('\n\nparm values for model=%s topo=(((1,4),3),2) mug=%.3e  seqlen=%.4e\n' % (modelx,mug,seqlen))
	fw.write('theta_anc= %.3e\n' % thet_a)
	fw.write('theta_1= %.3e\n' % theta_1)
	fw.write('theta_4= %.3e\n' % theta_4)
	fw.write('theta_3= %.3e\n' % theta_3)
	fw.write('theta_2= %.3e\n' % theta_2)
	fw.write('F_1b(inbr.coef)= %.3e\n' % F_1b)
	fw.write('F_1c(inbr.coef)= %.3e\n' % F_1c)
	fw.write('F_1F(inbr.coef)= %.3e\n' % F_1F)
	fw.write('F_4(inbr.coef)= %.3e\n' % F_4)
	fw.write('F_3(inbr.coef)= %.3e\n' % F_3)
	fw.write('F_2(inbr.coef)= %.3e\n' % F_2)
	fw.write('N_anc= %.3e\n' % N_anc)
	fw.write('N_1b= %.3e\n' % N_1b)
	fw.write('N_1c= %.3e\n' % N_1c)
	fw.write('N_1F= %.3e\n' % N_1F)
	fw.write('N_4= %.3e\n' % N_4)
	fw.write('N_3= %.3e\n' % N_3)
	fw.write('N_2= %.3e\n' % N_2)
	fw.write('T2[gen]= %.3e\n' % T2)
	fw.write('T3[gen]= %.3e\n' % T3)
	fw.write('T4[gen]= %.3e\n' % T4)
	fw.write('T4+T3[gen]= %.3e\n' % (T4+T3))
	fw.write('T4+T3+T2[gen]= %.3e\n' % (T4+T3+T2))
	if modelx=='MPRS_8m':
		Mb2=popt[6]
		M2b=popt[7]
		Mc3=popt[8] 
		M3c=popt[9] 
		M13=popt[10]
		M31=popt[11]
		M14=popt[12]
		M41=popt[13]
		fw.write('Mb2(/2Nref)= %.3e\n' % Mb2)
		fw.write('M2b(/2Nref)= %.3e\n' % M2b)
		fw.write('Mc3(/2Nref)= %.3e\n' % Mc3)
		fw.write('M3c(/2Nref)= %.3e\n' % M3c)
		fw.write('Mig13(/2Nref)= %.3e\n' % M13)
		fw.write('Mig31(/2Nref)= %.3e\n' % M31)
		fw.write('Mig14(/2Nref)= %.3e\n' % M14)
		fw.write('Mig41(/2Nref)= %.3e\n' % M41)
	elif modelx=='MPRS_sym4m':
		Mb2=popt[6]
		Mc3=popt[7] 
		M13=popt[8]
		M14=popt[9]
		fw.write('symMb2(/2Nref)= %.3e\n' % Mb2)
		fw.write('symMc3(/2Nref)= %.3e\n' % Mc3)
		fw.write('symMig13(/2Nref)= %.3e\n' % M13)
		fw.write('symMig14(/2Nref)= %.3e\n' % M14)
elif modelx.startswith('MxPxRS_') or modelx.startswith('fMxPxRS_'):  # 
	N_1b=popt[0]*N_anc
	if modelx.startswith('fMxPxRS_'): 
	    f1C=popt[1]
	    N_1c=f1C*N_1b
	    f1F=popt[2]
	    N_1F=N_1c*f1F
	elif modelx.startswith('MxPxRS_'): 
	    N_1c=popt[1]*N_anc
	    N_1F=popt[2]*N_anc
	N_2b=popt[3]*N_anc
	f2X=popt[4]
	N_2F=N_2b*f2X
	N_3c=popt[5]*N_anc
	f3X=popt[6]
	N_3F=N_3c*f3X
	N_4=popt[7]*N_anc
	T2 =popt[-3]*2*N_anc
	T3 =popt[-2]*2*N_anc
	T4 =popt[-1]*2*N_anc
	fw.write('\n\nparm values for model=%s topo=(((1,4),3),2) mug=%.3e  seqlen=%.4e\n' % (modelx,mug,seqlen))
	fw.write('N_anc= %.3e\n' % N_anc)
	fw.write('N_1b= %.3e\n' % N_1b)
	if modelx.startswith('fMxPxRS_'): 
	    fw.write('N_1c= %.3e  f1C= %g\n' % (N_1c,f1C))
	    fw.write('N_1F= %.3e  f1F= %g\n' % (N_1F,f1F))
	elif modelx.startswith('MxPxRS_'): 
	    fw.write('N_1c= %.3e\n' % N_1c)
	    fw.write('N_1F= %.3e\n' % N_1F)
	fw.write('N_4= %.3e\n' % N_4)
	fw.write('N_3c= %.3e\n' % N_3c)
	fw.write('N_3F,f3X= %.3e  %g\n' % (N_3F,f3X))
	fw.write('N_2b= %.3e\n' % N_2b)
	fw.write('N_2F,f2X= %.3e  %g\n' % (N_2F,f2X))
	fw.write('T2[gen]= %.3e\n' % T2)
	fw.write('T3[gen]= %.3e\n' % T3)
	fw.write('T4[gen]= %.3e\n' % T4)
	fw.write('T4+T3[gen]= %.3e\n' % (T4+T3))
	fw.write('T4+T3+T2[gen]= %.3e\n' % (T4+T3+T2))
	if modelx=='MxPxRS_sym4m':
		Mb2=popt[8]
		Mc3=popt[9] 
		M13=popt[10]
		M14=popt[11]
		fw.write('symMb2(/2Nref)= %.3e\n' % Mb2)
		fw.write('symMc3(/2Nref)= %.3e\n' % Mc3)
		fw.write('symMig13(/2Nref)= %.3e\n' % M13)
		fw.write('symMig14(/2Nref)= %.3e\n' % M14)
elif modelx.startswith('MePeRS_'):  # nu1A,nu1B,nu1C,nu2A,nu2B,nu3B,nu3C, [M1b, M12, M13, M23],z1,z2,z3,T2, T3
	N_1b=popt[0]*N_anc
	N_1c=popt[1]*N_anc
	N_1F=popt[2]*N_anc
	N_2b=popt[3]*N_anc
	N_2F=popt[4]*N_anc
	N_3c=popt[5]*N_anc
	N_3F=popt[6]*N_anc
	N_4=popt[7]*N_anc
	T2 =popt[-3]*2*N_anc
	T3 =popt[-2]*2*N_anc
	T4 =popt[-1]*2*N_anc
	fw.write('\n\nparm values for model=%s topo=(((1,4),3),2) mug=%.3e  seqlen=%.4e\n' % (modelx,mug,seqlen))
	fw.write('N_anc= %.3e\n' % N_anc)
	fw.write('N_1b= %.3e\n' % N_1b)
	fw.write('N_1c= %.3e\n' % N_1c)
	fw.write('N_1F= %.3e\n' % N_1F)
	fw.write('N_4= %.3e\n' % N_4)
	fw.write('N_3c= %.3e\n' % N_3c)
	fw.write('N_3F= %.3e\n' % N_3F)
	fw.write('N_2b= %.3e\n' % N_2b)
	fw.write('N_2F= %.3e\n' % N_2F)
	fw.write('T2[gen]= %.3e\n' % T2)
	fw.write('T3[gen]= %.3e\n' % T3)
	fw.write('T4[gen]= %.3e\n' % T4)
	fw.write('T4+T3[gen]= %.3e\n' % (T4+T3))
	fw.write('T4+T3+T2[gen]= %.3e\n' % (T4+T3+T2))
	if modelx=='MePeRS_sym4m':
		Mb2=popt[8]
		Mc3=popt[9] 
		M13=popt[10]
		M14=popt[11]
		fw.write('symMb2(/2Nref)= %.3e\n' % Mb2)
		fw.write('symMc3(/2Nref)= %.3e\n' % Mc3)
		fw.write('symMig13(/2Nref)= %.3e\n' % M13)
		fw.write('symMig14(/2Nref)= %.3e\n' % M14)


###### pop1 marginal AFS
AFS1d_D=dict()
AFS1d_M=dict()
idx_af=dict()
for i in range(npop):
    margi=list(set(range(npop))-set([i]))
    AFS1d_D[i]=masked_data.marginalize(margi)
    AFS1d_M[i]=masked_model.marginalize(margi)
    AFS1d_dev=(AFS1d_M[i]-AFS1d_D[i])/AFS1d_M[i]
    idx_af[i]=ma.array(data=numpy.arange(ns[i]+1),mask=ma.getmaskarray(AFS1d_M[i]))
    af1d = ma.vstack((idx_af[i].compressed(),AFS1d_D[i].compressed(),AFS1d_M[i].compressed(),AFS1d_dev.compressed())).transpose()
    af1d = numpy.array(af1d)
    fw.write('\n\n#AFS_pop%d  data model dev=(model-data)/model:\n%s\n' % (i+1,numpy.array2string(af1d,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']','')))
    ## rate
    S1_rate_data = AFS1d_D[i].sum()/seqlen
    S1_rate_model = AFS1d_M[i].sum()/seqlen
    fw.write('#SS_rate_pop%d:  dat= %.3e model= %.3e\n' % (i+1,S1_rate_data, S1_rate_model))
    thetaW_dat = AFS1d_D[i].Watterson_theta()/seqlen
    thetaW_model = AFS1d_M[i].Watterson_theta()/seqlen
    pi_dat = AFS1d_D[i].pi()/seqlen
    pi_model = AFS1d_M[i].pi()/seqlen
    TajimaD_dat = AFS1d_D[i].Tajima_D()/seqlen
    TajimaD_model = AFS1d_M[i].Tajima_D()/seqlen
    fw.write('#thetaWat_pop%d: dat= %.3e model= %.3e\n' % (i+1,thetaW_dat, thetaW_model))
    fw.write('#pi_pop%d:       dat= %.3e model= %.3e\n' % (i+1,pi_dat, pi_model))
    fw.write('#TajimaD_pop%d:  dat= %.3e model= %.3e\n' % (i+1,TajimaD_dat, TajimaD_model))


###### 2d AFS marginal
AFS2d_D=dict()
AFS2d_M=dict()
SS4_D=dict()
SS4_M=dict()
ratio_SS4=dict()
for i in range(npop-1):
    for j in range(i+1,npop):
        margi=list(set(range(npop))-set([i,j]))
        AFS2d_M[(i,j)]=masked_model.marginalize(margi)
        AFS2d_D[(i,j)]=masked_data.marginalize(margi)
        SS4_m = SS4_2pop(AFS2d_M[(i,j)])
        SS4_d = SS4_2pop(AFS2d_D[(i,j)])
        SS4_D[(i,j)] = SS4_d
        SS4_M[(i,j)] = SS4_m
        ratio_SS4[(i,j)] = SS4_m/SS4_d
        dev_SS4 = (SS4_m - SS4_d)/SS4_m
        matSS4=numpy.vstack((SS4_d,SS4_m,dev_SS4)).transpose()
        fw.write('\n\n#SS4[Sfx|Ssh|Sx1|Sx2]/pop%d%d: data  model  dev(M-D)/M\n%s\n' % (i+1,j+1,numpy.array2string(matSS4,max_line_width=200,precision=3,separator='\t').replace('[','').replace(']','')))
        Fst_data=AFS2d_D[(i,j)].Fst()
        Fst_model=AFS2d_M[(i,j)].Fst()
        fw.write('#Weir Fst(pop%d,pop%d): data= %.03f  model= %0.3f\n' % (i+1,j+1,Fst_data,Fst_model))



fw.write('#time passed %.1f sec\n' % ( time.time() - time0))
fw.close()
sys.exit(1)

