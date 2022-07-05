import argparse
import yaml
from math import ceil

empty_val = "NANANANA"

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
        "voxelArrayShift":0.0}
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",dest="input_path",type=str)
    parser.add_argument("--no_scale",dest="no_scale",action="store_true")
    args = parser.parse_args()

    settings_dict = settings_dict_template.copy()
    if args.no_scale == True:
        settings_dict["setting"]["normalize"] = False
    with open(args.input_path,'r') as o:
        for line in o:
            line = line.strip().split(",")
            if line[0] == "bw" and line[2] == "q0.5":
                bw = float(line[-1])
                if line[1] in settings_dict["imageType"]:
                    settings_dict["imageType"][line[1]]["binWidth"] = bw
                else:
                    settings_dict["imageType"][line[1]] = {"binWidth":bw}
                if line[1] == "Original":
                    settings_dict["setting"]["binWidth"] = bw
                if line[1] == "LoG":
                    settings_dict["imageType"]["LoG"]["sigma"] = [1]
            elif line[0] == "q01":
                m = float(line[-1])
                if m < 0:
                    m = ceil(-m)
                else:
                    m = 0
                if line[1] in settings_dict["imageType"]:
                    settings_dict["imageType"][line[1]]["voxelArrayShift"] = m
                else:
                    settings_dict["imageType"][line[1]] = {"voxelArrayShift":m}
                if line[1] == "Original":
                    settings_dict["setting"]["voxelArrayShift"] = m
    
    x = yaml.dump(settings_dict,indent=4)
    x = x.replace(empty_val,"")
    print(x)
