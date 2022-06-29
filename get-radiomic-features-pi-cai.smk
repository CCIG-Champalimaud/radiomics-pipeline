import os
import re
from glob import glob 

def shift(l,idxs):
    l = [l[i] for i in idxs]
    return l

def scale(wc):
    if wc.mod in ["adc"]:
        o = "--no_scale"
    else:
        o = ""
    return o

input_path = config["input_path"]
output_paths = {
    "radiomic_features": config["radiomic_features_path"],
    "dataset_information": config["dataset_information_path"],
    "aggregated_features": config["aggregated_features_path"],
    "masks": config["masks_path"],
    "radiomic_settings": config["radiomic_settings_path"]}
dataset_id = config["dataset_id"]
patterns = config["patterns"]
pattern_mask = config["mask_pattern"]
registration = config["registration"]

# optional arguments
n_bins = 100
if 'nbins' in config:
    n_bins = config['n_bins']

for k in output_paths:
    if k != "aggregated_features":
        os.makedirs(output_paths[k],exist_ok=True)
    else:
        os.makedirs(
            os.sep.join(output_paths[k].split(os.sep)[:-1]),exist_ok=True)

mask_dict = {}
for path in glob(os.path.join(output_paths["masks"],pattern_mask)):
    identifier = path.split(os.sep)[-1].split('.')[0]
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
        "{}/voxel_features.{}.{}.csv".format(output_paths["dataset_information"],k,dataset_id))
    output_radiomic_settings.append(
        "{}/config-{}.yaml".format(output_paths["radiomic_settings"],k))
    for path in glob(os.path.join(input_path,patterns[k])):
        o = path.replace(input_path,"")
        sub_o = '_'.join(o.split(os.sep)[-1].split('_')[:2])
        out_path = os.path.join(output_paths["radiomic_features"],sub_o+".json")
        sub_folder = o[0]
        if sub_o in mask_dict:
            correspondence_dict_masks[sub_o] = mask_dict[sub_o]
            if sub_o in correspondence_dict:
                correspondence_dict[sub_o].append(path)
            else:
                output_radiomics.append(out_path)
                correspondence_dict[sub_o] = [path]

output_aggregated = os.path.join(output_paths["aggregated_features"])

rule all:
    input:
        output_spacing,
        output_radiomics,
        output_aggregated,
        output_voxel_features,
        output_radiomic_settings

rule get_voxel_features:
    input:
        input_path
    output:
        "dataset_information/voxel_features.{mod}."+dataset_id+".csv"
    shell:
        """
        python3 utils/get-voxel-bins.py \
            --input_dir {input} \
            --pattern */*/*_{wildcards.mod}.mha \
            --rule minmax \
            --nbins 100 \
            --n_workers 8 > {output}
        """

rule get_radiomic_settings:
    input:
        "dataset_information/voxel_features.{mod}."+dataset_id+".csv"
    output:
        "config-feature-extraction/config-{mod}.yaml"
    params:
        scale=scale
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
        mod_spacing=list(patterns.keys())[0]
    shell:
        """
        python3 utils/extract-radiomic-features.py \
            --input_paths {input.files} \
            --configs {params.configs} \
            --mask_path {input.mask} \
            --target_spacing $(cat {params.di_path}/spacing.{params.mod_spacing}.{dataset_id} | tr ',' ' ') \
            --registration {registration} \
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
            --input_paths {input} > {output}
        """
