import os
import re
from glob import glob 

def shift(l,idxs):
    l = [l[i] for i in idxs]
    return l

input_path = "../../data/PI-CAI/dataset_resized_corrected"
output_paths = {
    "radiomic_features":"../../data/PI-CAI/radiomic_features",
    "dataset_information":"dataset_information",
    "aggregated_features":"radiomic_features",
    "masks":"../../data/PI-CAI/labels/csPCa_lesion_delineations/human_expert/original"}

config_files = [
    "config-feature-extraction/config-t2w.yaml",
    "config-feature-extraction/config-adc.yaml",
    "config-feature-extraction/config-dwi.yaml"]

for k in output_paths:
    os.makedirs(output_paths[k],exist_ok=True)

patterns = {
    "T2W":"*/*/*_t2w.mha",
    "ADC":"*/*/*_adc.mha",
    "HBV":"*/*/*_hbv.mha"}
pattern_mask = "*/*nii.gz"

mask_dict = {}
for path in glob(os.path.join(output_paths["masks"],pattern_mask)):
    identifier = path.split(os.sep)[-1].split('.')[0]
    mask_dict[identifier] = path

output_spacing = []
correspondence_dict = {}
correspondence_dict_masks = {}
output_radiomics = []
for k in patterns:
    output_spacing.append(
        "{}/spacing.{}.PICAI".format(output_paths["dataset_information"],k))
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

output_aggregated = os.path.join(output_paths["aggregated_features"],"features.csv")

rule all:
    input:
        output_spacing,output_radiomics,
        output_aggregated

rule get_spacing:
    input:
        input_path
    output:
        os.path.join(output_paths["dataset_information"],"spacing.{mod}.PICAI")
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
    output:
        os.path.join(output_paths["radiomic_features"],"{identifier}.json")
    params:
        configs=" ".join(config_files),
        di_path=output_paths["dataset_information"]
    shell:
        """
        python3 utils/extract-radiomic-features.py \
            --input_paths {input.files} \
            --configs {params.configs} \
            --mask_path {input.mask} \
            --target_spacing $(cat {params.di_path}/spacing.T2W.PICAI | tr ',' ' ') \
            --registration first \
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
