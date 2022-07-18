import os
import re
from glob import glob 

input_path = config["input_path"]
masks_path = config["masks_path"]
output_paths = {
    "radiomic_features": config["radiomic_features_path"],
    "dataset_information": config["dataset_information_path"],
    "radiomic_settings": config["radiomic_settings_path"]}
dataset_id = config["dataset_id"]
patterns = config["patterns"]
pattern_mask = config["mask_pattern"]
registration = config["registration"]
id_pattern = config["id_pattern"]

# optional arguments
n_bins = 100
no_scale_keys = ["adc","ADC"]
conditional_multiplication = {"adc":[1000,0.001]}
if 'n_bins' in config:
    n_bins = config['n_bins']
if 'no_scale_keys' in config:
    no_scale_keys = config['no_scale_keys']
if 'cond_mult' in config:
    conditional_multiplication = config['cond_mult']

def shift(l,idxs):
    l = [l[i] for i in idxs]
    return l

def no_scale(wc):
    if wc.mod in no_scale_keys: o = "--no_scale"
    else: o = ""
    return o

def scale(wc):
    if wc.mod in no_scale_keys: o = ""
    else: o = "--scale"
    return o

def cond_mult(wc):
    if wc.mod in conditional_multiplication:
        m = conditional_multiplication[wc.mod]
        o = "--conditional_multiplication {} {}".format(*m)
    else: 
        o = ""
    return o

def cond_mult_abs(wc):
    if len(conditional_multiplication) > 0:
        K = list(conditional_multiplication.keys())[0]
        m = conditional_multiplication[K]
        o = "--conditional_multiplication {} {}".format(*m)
    else: 
        o = ""
    return o

def cond_mult_idx(wc):
    if len(conditional_multiplication) > 0:
        K = list(conditional_multiplication.keys())
        P = list(patterns.keys())
        if K[0] in P:
            idx = P.index(K[0])
            o = "--cond_mult_idx " + str(idx)
        else:
            o = ""
    else: 
        o = ""
    return o

for k in output_paths:
    os.makedirs(output_paths[k],exist_ok=True)

mask_dict = {}
for path in glob(os.path.join(masks_path,pattern_mask)):
    identifier = re.search(id_pattern,path).group()
    mask_dict[identifier] = path

output_spacing = []
output_voxel_features = []
output_radiomic_settings = []
correspondence_dict = {}
correspondence_dict_masks = {}
output_radiomics = []
for k in patterns:
    output_spacing.append(
        "{}/spacing.{}.{}".format(output_paths["dataset_information"],k,dataset_id))
    output_voxel_features.append(
        "{}/voxel_features.{}.{}.{}.csv".format(
            output_paths["dataset_information"],k,dataset_id,n_bins))
    output_radiomic_settings.append(
        os.path.join(output_paths["radiomic_settings"],"config-{}-{}-{}.yaml").format(
            k,n_bins,dataset_id
        ))
    for path in glob(os.path.join(input_path,patterns[k])):
        o = path.replace(input_path,"")
        sub_o = re.search(id_pattern,o).group()
        out_path = os.path.join(output_paths["radiomic_features"],sub_o+".json")
        sub_folder = o[0]
        if sub_o in mask_dict:
            correspondence_dict_masks[sub_o] = mask_dict[sub_o]
            if sub_o in correspondence_dict:
                correspondence_dict[sub_o].append(path)
            else:
                output_radiomics.append(out_path)
                correspondence_dict[sub_o] = [path]

rule all:
    input:
        output_spacing,
        output_radiomics,
        output_voxel_features,
        output_radiomic_settings

rule get_voxel_features:
    input:
        input_path
    output:
        os.path.join(
            output_paths["dataset_information"],
            "voxel_features.{mod}." + dataset_id + "." + str(n_bins) + ".csv")
    params:
        scale=scale,
        cond_mult=cond_mult,
        pattern=lambda wc: patterns[wc.mod]
    shell:
        """
        python3 utils/get-voxel-bins.py \
            --input_dir {input} \
            --pattern {params.pattern} \
            --rule minmax \
            --nbins {n_bins} \
            --n_workers 8 \
            --output_paths {output} \
            {params.scale} {params.cond_mult} \
            --exclude_transforms LBP2D
        """

rule get_radiomic_settings:
    input:
        os.path.join(
            output_paths["dataset_information"],
            "voxel_features.{mod}."+dataset_id+"." + str(n_bins) + ".csv")
    output:
        os.path.join(output_paths["radiomic_settings"],"config-{mod}-{n_bins}-{dataset_id}.yaml")
    params:
        scale=no_scale
    shell:
        """
        python3 utils/voxel-features-to-radiomics-settings.py \
            --input_path {input} {params.scale} > {output}
        """

rule get_spacing:
    input:
        input_path
    output:
        os.path.join(output_paths["dataset_information"],"spacing.{mod}."+dataset_id)
    params:
        pattern=lambda wc: patterns[wc.mod],
        q=0.9,
        parameter="spacing"
    shell:
        """
        python3 utils/get-info.py \
    	    --input_dir {input_path} \
    	    --pattern {params.pattern} \
    	    --parameter {params.parameter} \
    	    --quantile {params.q} > {output}
        """

rule get_radiomic_features:
    input:
        files=lambda wc: correspondence_dict[wc.identifier],
        mask=lambda wc: correspondence_dict_masks[wc.identifier],
        radiomic_settings=output_radiomic_settings
    output:
        os.path.join(output_paths["radiomic_features"],"{identifier}.json")
    params:
        configs=" ".join(output_radiomic_settings),
        di_path=output_paths["dataset_information"],
        mod_spacing=list(patterns.keys())[0],
        cond_mult=cond_mult_abs,
        cond_mult_idx=cond_mult_idx
    shell:
        """
        python3 utils/extract-radiomic-features.py \
            --input_paths {input.files} \
            --configs {params.configs} \
            --mask_path {input.mask} \
            --target_spacing $(cat {params.di_path}/spacing.{params.mod_spacing}.{dataset_id} | tr ',' ' ') \
            --registration {registration} \
            {params.cond_mult} {params.cond_mult_idx} \
            --output_path {output}
        """
