import argparse
import os
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from radiomics import imageoperations,setVerbosity
from glob import glob
from tqdm import tqdm

setVerbosity(60)

desc = """
Determines the number of necessary bins to describe the pixel range in a 
given set of MRI scans using different rules. Specifying multiple nbins
and output_paths is only valid for rule "minmax".
"""

transform_factory = {
    "Original": imageoperations.getOriginalImage,
    "Wavelet": imageoperations.getExponentialImage,
    "LoG": imageoperations.getLoGImage,
    "Square": imageoperations.getSquareImage,
    "SquareRoot": imageoperations.getSquareImage,
    "Logarithm": imageoperations.getLogarithmImage,
    "Exponential": imageoperations.getExponentialImage,
    "Gradient": imageoperations.getGradientImage,
    "LBP2D": imageoperations.getLBP2DImage}

def print_write(x,o):
    o.write(x+'\n')
    print(x)

def fd(data):
    iqr = np.quantile(data,0.75) - np.quantile(data,0.25)
    bw = 2 * iqr / (data.size**(1/3))
    m,M = data.min(),data.max()
    nbins = (M,m)/bw
    q01,q99 = np.quantile(data,[0.01,0.99])
    return nbins,bw,m,M,q01,q99

def sturgis(data):
    nbins = 1 + np.log2(data.size)
    m,M = data.min(),data.max()
    bw = (M-m)/nbins
    q01,q99 = np.quantile(data,[0.01,0.99])
    return nbins,bw,m,M,q01,q99

def minmax(data,nbins):
    m,M = data.min(),data.max()
    bw = (M - m)/nbins
    q01,q99 = np.quantile(data,[0.01,0.99])
    return nbins,bw,m,M,q01,q99

def unbox(l):
    unboxed = False
    while unboxed == False:
        if isinstance(l,list) or isinstance(l,tuple):
            l = l[0]
        else:
            unboxed = True
    return l

def wrapper(sitk_image,transforms,k,rule,nbins):
    t = transforms[k]
    if k == "LoG":
        transformed_image = unbox(list(t(sitk_image,None,sigma=[1])))
    else:
        transformed_image = unbox(list(t(sitk_image,None)))
    d = sitk.GetArrayFromImage(transformed_image).ravel()
    if rule == "fd":
        nbins = nbins[0]
        nbins,bw,m,M,q01,q99 = fd(d)
    elif rule == "sturgis":
        nbins = nbins[0]
        nbins,bw,m,M,q01,q99 = sturgis(d)
    elif rule == "minmax":
        Out = [],[],[],[],[],[]
        for nb in nbins:
            out = minmax(d,nb)
            for o,O in zip(out,Out):
                O.append(o)
        nbins,bw,m,M,q01,q99 = Out
    return k,(nbins,bw,m,M,q01,q99)

def wrapper_map(x):
    return wrapper(*x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--input_dir',dest='input_dir')
    parser.add_argument(
        '--output_paths',dest='output_paths',type=str,nargs='+')
    parser.add_argument(
        '--pattern',dest='pattern',default="*nii.gz")
    parser.add_argument(
        '--rule',dest="rule",type=str,default="fd",
        choices=["fd","sturgis","minmax"])
    parser.add_argument(
        '--nbins',dest="nbins",type=int,default=20,nargs="+")
    parser.add_argument(
        '--scale',dest="scale",action="store_true",default=False)
    parser.add_argument(
        '--transform',dest="transform",default="all",
        choices=["all",*transform_factory.keys()])
    parser.add_argument(
        '--exclude_transforms',dest="exclude_transforms",default=None,
        choices=[*transform_factory.keys()],nargs='+',type=str)
    parser.add_argument(
        '--n_workers',dest='n_workers',type=int,
        default=1)
    parser.add_argument(
        '--conditional_multiplication',dest='conditional_multiplication',
        type=int,nargs=2,default=None)

    args = parser.parse_args()

    if args.n_workers > 1:
        pool = Pool(args.n_workers)

    if args.transform == "all":
        transforms = transform_factory
    else:
        transforms = {args.transform:transform_factory[args.transform]}
    if args.exclude_transforms is not None:
        transforms = {k:transforms[k] for k in transforms
                      if k not in args.exclude_transforms}

    all_data = {
        N:{
            k:{"nbins":[],"bw":[],"min":[],"max":[],"q01":[],"q99":[]}
            for k in transforms}
        for N in args.nbins}
    for path in tqdm(glob(os.path.join(args.input_dir,args.pattern))):
        sitk_image = sitk.ReadImage(path)
        if args.conditional_multiplication is not None:
            am = args.conditional_multiplication
            M = sitk.GetArrayFromImage(sitk_image).max()
            if M > am[0]:
                sitk_image = sitk.Multiply(
                    sitk.Cast(sitk_image,sitk.sitkFloat32),am[1])
        if args.scale == True:
            sitk_image = imageoperations.normalizeImage(sitk_image)
        if args.n_workers > 1:
            x = []
            for k in transforms:
                x.append([
                    sitk_image,transforms,k,args.rule,args.nbins])
            output = pool.map(wrapper_map,x)
        else:
            def partial_wrapper(k):
                return wrapper(
                    sitk_image,transforms,k,args.rule,args.nbins)
            output = map(
                partial_wrapper,transforms.keys())
        if args.rule == "minmax":
            for k,(nbins,bw,m,M,q01,q99) in output:
                for i,N in enumerate(args.nbins):
                    all_data[N][k]["nbins"].append(nbins[i])
                    all_data[N][k]["bw"].append(bw[i])
                    all_data[N][k]["min"].append(m[i])
                    all_data[N][k]["max"].append(M[i])
                    all_data[N][k]["q01"].append(q01[i])
                    all_data[N][k]["q99"].append(q99[i])
        else:
            N = args.nbins[0]
            for k,(nbins,bw,m,M,q01,q99) in output:
                all_data[N][k]["nbins"].append(nbins)
                all_data[N][k]["bw"].append(bw)
                all_data[N][k]["min"].append(m)
                all_data[N][k]["max"].append(M)
                all_data[N][k]["q01"].append(q01)
                all_data[N][k]["q99"].append(q99)

    quant = [0,0.05,0.5,0.95,1.0]
    for N,p in zip(all_data,args.output_paths):
        output = open(p,'w')
        for k in all_data[N]:
            bw = np.quantile(all_data[N][k]["bw"],quant)
            bw_mean = np.mean(all_data[N][k]["bw"])
            nbins = np.quantile(all_data[N][k]["nbins"],quant)
            m = np.min(all_data[N][k]["min"])
            M = np.max(all_data[N][k]["max"])
            q01 = np.min(all_data[N][k]["q01"])
            q99 = np.max(all_data[N][k]["q99"])

            for b in zip(bw,quant):
                print_write("bw,{},q{},{}".format(k,b[1],b[0]),output)
            print_write("bw,{},mean,{}".format(k,bw_mean),output)
            for n in zip(nbins,quant):
                print_write(
                    "nbin_dist,{},{},{}".format(k,n[1],n[0]),output)
            print_write("min,{},0,{}".format(k,m),output)
            print_write("max,{},1,{}".format(k,M),output)
            print_write("q01,{},0,{}".format(k,q01),output)
            print_write("q99,{},1,{}".format(k,q99),output)