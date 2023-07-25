import os
import re
from glob import glob 

input_path = config["input_path"]
masks_path = config["masks_path"]
output_paths = {
    "radiomic_features": config["radiomic_features_path"],
    "dataset_information": config["dataset_information_path"],
    "aggregated_features": config["aggregated_features_path"],
    "radiomic_settings": config["radiomic_settings_path"]}
dataset_id = config["dataset_id"]
patterns = config["patterns"]
pattern_mask = config["mask_pattern"]
registration = config["registration"]
id_pattern = config["id_pattern"]

# optional arguments
opt_args = {
    "n_bins": 100,
    "no_scale_keys": [],
    "conditional_multiplication": {},
    "additional_arguments":"",
    "transforms":["all"],
    "features":["all"],
    "additional_settings":[]
}
for k in opt_args:
    if k in config:
        opt_args[k] = config[k]
n_bins = opt_args["n_bins"]
no_scale_keys = opt_args["no_scale_keys"]
conditional_multiplication = opt_args["conditional_multiplication"]
additional_arguments = opt_args["additional_arguments"]
transforms = opt_args["transforms"]
features = opt_args["features"]

if "minimumROISize" in additional_arguments:
    minimum_size = int([x.split("=")[1] for x in additional_arguments.split(" ")
                        if "minimumROISize" in x][0])
else:
    minimum_size = 2

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
        idx = list(patterns.keys()).index(K[0])
        o = "--cond_mult_idx " + str(idx)
    else: 
        o = ""
    return o

def get_additional_settings():
    if len(config["additional_settings"]) > 0:
        return f"--additional_settings {' '.join(config['additional_settings'])}"
    else:
        return ""

for k in output_paths:
    if k != "aggregated_features":
        os.makedirs(output_paths[k],exist_ok=True)
    else:
        d = os.sep.join(output_paths[k].split(os.sep)[:-1])
        if len(d) > 0:
            os.makedirs(d,exist_ok=True)

mask_dict = {}
for path in glob(os.path.join(masks_path,pattern_mask)):
    identifier = re.search(id_pattern,path).group()
    mask_dict[identifier] = path

output_spacing = []
output_voxel_features = []
output_radiomic_settings = {str(bin_number):[] for bin_number in n_bins}
correspondence_dict = {str(bin_number):{} for bin_number in n_bins}
correspondence_dict_masks = {str(bin_number):{} for bin_number in n_bins}
output_radiomics = {str(bin_number):[] for bin_number in n_bins}
for k in patterns:
    for bin_number in n_bins:
        bin_number = str(bin_number)
        output_spacing.append(
            "{}/spacing.{}.{}".format(output_paths["dataset_information"],k,dataset_id))
        output_voxel_features.append(
            "{}/voxel_features.{}.{}_{}.json".format(
                output_paths["dataset_information"],k,dataset_id,bin_number))
        output_radiomic_settings[bin_number].append(
            os.path.join(
                output_paths["radiomic_settings"],
                "config-{}-{}-{}.yaml").format(
                    k,bin_number,dataset_id))
        for path in glob(os.path.join(input_path,patterns[k])):
            o = path.replace(input_path,"")
            sub_o = re.search(id_pattern,o).group()
            out_path = os.path.join(output_paths["radiomic_features"],f"{sub_o}.{bin_number}.json")
            sub_folder = o[0]
            if sub_o in mask_dict:
                correspondence_dict_masks[bin_number][sub_o] = mask_dict[sub_o]
                if sub_o in correspondence_dict[bin_number]:
                    correspondence_dict[bin_number][sub_o].append(path)
                else:
                    output_radiomics[bin_number].append(out_path)
                    correspondence_dict[bin_number][sub_o] = [path]

output_aggregated = []
for bin_number in n_bins:
    os.path.join(
        output_paths["aggregated_features"],
        f"{dataset_id}_{'.'.join(patterns.keys())}_{bin_number}.csv")

rule all:
    input:
        output_spacing,
        [output_radiomics[k] for k in output_radiomics],
        output_aggregated,
        output_voxel_features,
        [output_radiomic_settings[k] for k in output_radiomic_settings]

rule get_voxel_features:
    input:
        input_path
    output:
        [os.path.join(
            output_paths["dataset_information"],
            "voxel_features.{mod}."+dataset_id+ "_" + str(n) + ".json")
        for n in n_bins]
    params:
        scale=scale,
        cond_mult=cond_mult,
        pattern=lambda wc: patterns[wc.mod],
        n_workers=workflow.cores
    threads:
        workflow.cores
    shell:
        """
        python3 utils/get-all-info.py \
            --input_dir {input} \
            --pattern {params.pattern} \
            --rule minmax \
            --nbins {n_bins} \
            --n_workers {params.n_workers} \
            --output_paths {output} \
            {params.scale} {params.cond_mult} \
            --transforms {transforms}
        """

rule get_radiomic_settings:
    input:
        os.path.join(
            output_paths["dataset_information"],
            "voxel_features.{mod}." + dataset_id + "_{n_bins}.json")
    output:
        os.path.join(output_paths["radiomic_settings"],"config-{mod}-{n_bins}-{dataset_id}.yaml")
    params:
        scale=no_scale,
        additional_settings=get_additional_settings(),
    shell:
        """
        python3 utils/voxel-features-to-radiomics-settings.py \
            --input_path {input} \
            {params.scale} \
            {params.additional_settings} > {output}
        """

rule get_spacing:
    input:
        [os.path.join(
            output_paths["dataset_information"],
            "voxel_features.{mod}."+dataset_id+ "_" + str(n) + ".json")
        for n in n_bins][0]
    output:
        os.path.join(output_paths["dataset_information"],"spacing.{mod}."+dataset_id)
    params:
        q=0.5,
        parameter="spacing"
    shell:
        """
        python3 utils/summarise-info.py \
    	    --input_path {input} \
    	    --parameter {params.parameter} \
    	    --quantile {params.q} > {output}
        """

rule get_radiomic_features:
    input:
        files = lambda wc: correspondence_dict[wc.n_bins][wc.identifier],
        mask = lambda wc: correspondence_dict_masks[wc.n_bins][wc.identifier],
        radiomic_settings = lambda wc: output_radiomic_settings[wc.n_bins],
        spacing = lambda wc: os.path.join(
            output_paths["dataset_information"],
            f"spacing.{list(patterns.keys())[0]}." + dataset_id),
        configs=lambda wc: " ".join(output_radiomic_settings[wc.n_bins]),
    output:
        os.path.join(output_paths["radiomic_features"],"{identifier}.{n_bins}.json")
    params:
        di_path=output_paths["dataset_information"],
        mod_spacing=list(patterns.keys())[0],
        cond_mult=cond_mult_abs,
        cond_mult_idx=cond_mult_idx
    shell:
        """
        python3 utils/extract-radiomic-features.py \
            --input_paths {input.files} \
            --configs {input.configs} \
            --mask_path {input.mask} \
            --target_spacing $(cat {params.di_path}/spacing.{params.mod_spacing}.{dataset_id} | tr ',' ' ') \
            --registration {registration} \
            {params.cond_mult} {params.cond_mult_idx} \
            --minimum_lesion_size {minimum_size} \
            --output_path {output}
        """

rule aggregate_features:
    input:
        output_radiomics
    output:
        output_aggregated
    shell:
        """
        python3 utils/aggregate-features.py \
            --input_paths {input} --id_pattern {id_pattern} > {output}
        """
