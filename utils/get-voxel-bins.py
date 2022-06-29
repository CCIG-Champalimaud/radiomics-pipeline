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
given set of MRI scans using different rules.
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
        nbins,bw,m,M,q01,q99 = fd(d)
    elif rule == "sturgis":
        nbins,bw,m,M,q01,q99 = sturgis(d)
    elif rule == "minmax":
        nbins,bw,m,M,q01,q99 = minmax(d,nbins)
    return k,(nbins,bw,m,M,q01,q99)

def wrapper_map(x):
    return wrapper(*x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--input_dir',dest='input_dir')
    parser.add_argument(
        '--pattern',dest='pattern',default="*nii.gz")
    parser.add_argument(
        '--rule',dest="rule",type=str,default="fd",choices=["fd","sturgis","minmax"])
    parser.add_argument(
        '--nbins',dest="nbins",type=int,default=20)
    parser.add_argument(
        '--scale',dest="scale",action="store_true",default=False)
    parser.add_argument(
        '--transform',dest="transform",default="all",
        choices=["all",*transform_factory.keys()])
    parser.add_argument(
        '--n_workers',dest='n_workers',type=int,
        default=1)
    args = parser.parse_args()

    if args.n_workers > 1:
        pool = Pool(args.n_workers)

    if args.transform == "all":
        transforms = transform_factory
    else:
        transforms = {args.transform:transform_factory[args.transform]}

    all_data = {k:{"nbins":[],"bw":[],"min":[],"max":[],"q01":[],"q99":[]}
                for k in transforms}
    for path in tqdm(glob(os.path.join(args.input_dir,args.pattern))):
        sitk_image = sitk.ReadImage(path)
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
        for k,(nbins,bw,m,M,q01,q99) in output:
            all_data[k]["nbins"].append(nbins)
            all_data[k]["bw"].append(bw)
            all_data[k]["min"].append(m)
            all_data[k]["max"].append(M)
            all_data[k]["q01"].append(q01)
            all_data[k]["q99"].append(q99)
    
    quant = [0,0.05,0.5,0.95,1.0]
    for k in all_data:
        bw = np.quantile(all_data[k]["bw"],quant)
        bw_mean = np.mean(all_data[k]["bw"])
        nbins = np.quantile(all_data[k]["nbins"],quant)
        m = np.min(all_data[k]["min"])
        M = np.max(all_data[k]["max"])
        q01 = np.min(all_data[k]["q01"])
        q99 = np.max(all_data[k]["q99"])

        for b in zip(bw,quant):
            print("bw,{},q{},{}".format(k,b[1],b[0]))
        print("bw,{},mean,{}".format(k,bw_mean))
        for n in zip(nbins,quant):
            print("nbin_dist,{},{},{}".format(k,n[1],n[0]))
        print("min,{},0,{}".format(k,m))
        print("max,{},1,{}".format(k,M))
        print("q01,{},0,{}".format(k,q01))
        print("q99,{},1,{}".format(k,q99))