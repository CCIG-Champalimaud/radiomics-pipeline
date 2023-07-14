import argparse
import os
import numpy as np
import SimpleITK as sitk
import json
from multiprocessing import Pool
from radiomics import imageoperations,setVerbosity
from glob import glob
from pathlib import Path
from tqdm import tqdm

from typing import List, Tuple, Any

setVerbosity(60)

desc = """
Determines the number of necessary bins to describe the pixel range in a 
given set of MRI scans using different rules. Specifying multiple nbins
and output_paths is only valid for rule "minmax".
"""

transform_factory = {
    "Original": imageoperations.getOriginalImage,
    "Wavelet": imageoperations.getWaveletImage,
    "LoG": imageoperations.getLoGImage,
    "Square": imageoperations.getSquareImage,
    "SquareRoot": imageoperations.getSquareImage,
    "Logarithm": imageoperations.getLogarithmImage,
    "Exponential": imageoperations.getExponentialImage,
    "Gradient": imageoperations.getGradientImage,
    # "LBP2D": imageoperations.getLBP2DImage
    }

def print_write(x,o):
    o.write(x+'\n')
    print(x)

def fd(data:np.ndarray)->Tuple[int,float,float,float,float]:
    iqr = np.quantile(data,0.75) - np.quantile(data,0.25)
    bw = 2 * iqr / (data.size**(1/3))
    m,M = float(data.min()),float(data.max())
    nbins = (M-m)/bw
    q01,q99 = np.quantile(data,[0.01,0.99]).tolist()
    return nbins,bw,m,M,q01,q99

def sturgis(data:np.ndarray)->Tuple[int,float,float,float,float]:
    nbins = 1 + np.log2(data.size)
    m,M = float(data.min()),float(data.max())
    bw = (M-m)/nbins
    q01,q99 = np.quantile(data,[0.01,0.99]).tolist()
    return nbins,bw,m,M,q01,q99

def minmax(data:np.ndarray,nbins:int)->Tuple[int,float,float,float,float]:
    m,M = float(data.min()),float(data.max())
    bw = (M - m)/nbins
    q01,q99 = np.quantile(data,[0.01,0.99]).tolist()
    return nbins,bw,m,M,q01,q99

def unbox(l:List[Any])->Any:
    unboxed = False
    while unboxed == False:
        if isinstance(l,list) or isinstance(l,tuple):
            l = l[0]
        else:
            unboxed = True
    return l

def wrapper(data,
            rule:str,
            nbins:int):
    if rule == "fd":
        nbins,bw,m,M,q01,q99 = fd(data)
    elif rule == "sturgis":
        nbins,bw,m,M,q01,q99 = sturgis(data)
    elif rule == "minmax":
        nbins,bw,m,M,q01,q99 = minmax(data,nbins)
    return nbins,bw,m,M,q01,q99

def wrapper_map(x):
    return wrapper(*x)

class Operator:
    def __init__(self,
                 conditional_multiplication:Tuple[float,float],
                 scale:bool,
                 transforms:List[str],
                 quantile_fn:str,
                 nbins:List[int]
                 ):
        self.conditional_multiplication = conditional_multiplication
        self.scale = scale
        self.transforms = transforms
        self.quantile_fn = quantile_fn
        self.nbins = nbins

        self.transform_list_ = transform_factory

    def correct_image(self,sitk_image:sitk.Image):
        if self.conditional_multiplication is not None:
            am = self.conditional_multiplication
            M = sitk.GetArrayFromImage(sitk_image).max()
            if M > am[0]:
                sitk_image = sitk.Multiply(
                    sitk.Cast(sitk_image,sitk.sitkFloat32),am[1])
        if self.scale == True:
            sitk_image = imageoperations.normalizeImage(sitk_image)
        return sitk_image

    def __call__(self,sitk_image_path:str):
        sitk_image = sitk.ReadImage(sitk_image_path)
        sitk_image = self.correct_image(sitk_image)
        output = {}
        output["spacing"] = sitk_image.GetSpacing()
        output["size"] = sitk_image.GetSize()
        output["origin"] = sitk_image.GetOrigin()
        output["bin_width_info"] = {}
        for transform_str in self.transforms:
            transform = self.transform_list_[transform_str]
            if transform_str == "LoG":
                transformed_image = unbox(
                    list(transform(sitk_image,None,sigma=[1])))
            else:
                transformed_image = unbox(
                    list(transform(sitk_image,None)))
            data = sitk.GetArrayFromImage(transformed_image).ravel()
            for nbin in self.nbins:
                if nbin not in output["bin_width_info"]:
                    output["bin_width_info"][nbin] = {}
                output["bin_width_info"][nbin][transform_str] = wrapper(
                    data,self.quantile_fn,nbin)
                output["bin_width_info"][nbin][transform_str] = {
                    k:v for k,v in zip(
                    ["nbins","bandwidth","min","max","q01","q99"],
                    output["bin_width_info"][nbin][transform_str])}
        return sitk_image_path,output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--input_dir',dest='input_dir')
    parser.add_argument(
        '--output_paths',dest='output_paths',type=str,nargs='+',required=True)
    parser.add_argument(
        '--pattern',dest='pattern',default="*nii.gz")
    parser.add_argument(
        '--rule',dest="rule",type=str,default="fd",
        choices=["fd","sturgis","minmax"])
    parser.add_argument(
        '--nbins',dest="nbins",type=int,default=[20],nargs="+")
    parser.add_argument(
        '--scale',dest="scale",action="store_true",default=False)
    parser.add_argument(
        '--transforms',dest="transforms",default=["all"],
        choices=["all",*transform_factory.keys()],nargs="+")
    parser.add_argument(
        '--exclude_transforms',dest="exclude_transforms",default=None,
        choices=[*transform_factory.keys()],nargs='+',type=str)
    parser.add_argument(
        '--n_workers',dest='n_workers',type=int,default=1)
    parser.add_argument(
        '--conditional_multiplication',dest='conditional_multiplication',
        type=float,nargs=2,default=None)

    args = parser.parse_args()

    output_paths = []
    if len(args.output_paths) == 1 and len(args.nbins) > 1:
        output_path = args.output_paths[0]
        for nbin in args.nbins:
            tmp = output_path.split(".")
            output_base = ".".join(tmp[:-1])
            if len(tmp) > 1:
                extension = tmp[-1]
            else:
                extension = "json"
            new_output_path = "{}_{}.{}".format(output_base,nbin,extension)
            output_paths.append(new_output_path)
    elif len(args.output_paths) != len(args.nbins):
        raise Exception(
            "--output_paths should have length 1 or length identical to --nbins")
    else:
        output_paths = args.output_paths
    
    if args.n_workers > 1:
        pool = Pool(args.n_workers)

    if "all" in args.transforms[0]:
        transforms = transform_factory
    else:
        transforms = {k:transform_factory[k] for k in args.transforms}
    if args.exclude_transforms is not None:
        transforms = [k for k in transforms
                      if k not in args.exclude_transforms]

    all_paths = glob(os.path.join(args.input_dir,args.pattern))
    operator = Operator(
        conditional_multiplication=args.conditional_multiplication,
        scale=args.scale,
        transforms=list(transforms.keys()),
        quantile_fn=args.rule,
        nbins=args.nbins)
    if args.n_workers > 1:
        map_fn = Pool(args.n_workers).imap(operator,all_paths)
    else:
        map_fn = map(operator,all_paths)
    tmp_output = {}
    for output_path,output in tqdm(map_fn,total=len(all_paths),smoothing=0):
        tmp_output[output_path] = output
    
    for nbin,output_path in zip(args.nbins,output_paths):
        final_output = {}
        for k in tmp_output:
            final_output[k] = {}
            for feature_key in ["spacing","size","origin"]:
                final_output[k][feature_key] = tmp_output[k][feature_key]
            final_output[k]["bin_width_info"] = tmp_output[k]["bin_width_info"][nbin]
        Path(output_path).parent.mkdir(exist_ok=True,parents=True)
        with open(output_path,"w") as o:
            json.dump(final_output,o,indent=2)
