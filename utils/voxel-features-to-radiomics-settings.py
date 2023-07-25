import argparse
import yaml
import json
import numpy as np
from math import ceil

empty_val = "EMPTYVAL"

all_transforms = [
    "Original","Wavelet","LoG","Square","SquareRoot",
    "Logarithm","Exponential","Gradient","LBP2D"
]

settings_dict_template = {
    "imageType":{
        # LBP2D are positive integers, so it is the one filter for which
        # we can educatedly assume a reasonable binWidth
        "LBP2D":{
            "binWidth": 1.0,
            "voxelArrayShift": 0}
    },
    "featureClass":{
        "firstorder":empty_val,
        "glcm":empty_val,
        "glrlm":empty_val,
        "glszm":empty_val,
        "gldm":empty_val,
        "ngtdm":empty_val,
        "shape":empty_val},
    "setting":{
        "binWidth":None,
        "normalize":True,
        "normalizeScale":1,
        "force2D":True,
        "voxelArrayShift":0.0,
        "minimumROIDimensions":1}
}

all_feature_str = list(settings_dict_template["featureClass"].keys())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",dest="input_path",type=str)
    parser.add_argument(
        "--features",dest="features",default=["all"],nargs="+",type=str,
        choices=all_feature_str)
    parser.add_argument(
        "--transforms",dest="transforms",default=["all"],nargs="+",type=str,
        choices=all_transforms)
    parser.add_argument(
        "--no_scale",dest="no_scale",action="store_true")
    parser.add_argument(
        "--additional_settings",dest="additional_settings",nargs="+",
        help="Config for 'setting' tag. Must be provided as key1=v1 key2=v2.\
            Values are evaluated using eval")
    args = parser.parse_args()

    if "all" in args.features[0]:
        features = all_feature_str
    else:
        features = args.features

    if "all" in args.transforms[0]:
        transforms = all_transforms
    else:
        transforms = args.transforms

    for k_v in args.additional_settings:
        k,v = k_v.split("=")
        settings_dict_template["setting"][k] = eval(v)

    settings_dict = settings_dict_template.copy()
    settings_dict["featureClass"] = {
        k:settings_dict["featureClass"][k] 
        for k in settings_dict["featureClass"]
        if k in features
    }

    if args.no_scale == True:
        settings_dict["setting"]["normalize"] = False
    with open(args.input_path,'r') as o:
        data = json.loads(o.read())
        all_bw = {}
        all_q01 = {}
        for key in data:
            for transform_key in data[key]["bin_width_info"]:
                if transform_key not in all_bw:
                    all_bw[transform_key] = []
                    all_q01[transform_key] = []
                all_bw[transform_key].append(
                    data[key]["bin_width_info"][transform_key]["bandwidth"])
                all_q01[transform_key].append(
                    data[key]["bin_width_info"][transform_key]["bandwidth"])
        for transform_key in all_bw:
            bw = float(np.median(all_bw[transform_key]))
            if transform_key == "Original":
                settings_dict["setting"]["binWidth"] = bw
            if transform_key not in settings_dict:
                settings_dict["imageType"][transform_key] = {"binWidth":bw}
            if transform_key == "LoG":
                settings_dict["imageType"]["LoG"]["sigma"] = [1]

        for transform_key in all_q01:
            q01 = float(np.min(all_q01[transform_key]))
            if q01 < 0:
                q01 = ceil(-q01)
            else:
                q01 = 0
            if transform_key == "Original":
                settings_dict["setting"]["voxelArrayShift"] = q01
            settings_dict["imageType"][transform_key]["voxelArrayShift"] = q01
    
    x = yaml.dump(settings_dict,indent=4)
    x = x.replace(empty_val,"")
    print(x)
