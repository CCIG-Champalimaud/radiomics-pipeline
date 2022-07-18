import json
import argparse
import re
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.neighbors import KNeighborsClassifier

desc = """
Aggregates multiple json files and prints them in a comma separated format.
"""

def check_types(x,types):
    for t in types:
        if isinstance(x,t) == True:
            return True
    return False

def match_lesions(sub_df):
    k = "diagnostics_Mask-original_CenterOfMass"
    centers = np.array(
        [x.split(':') for x in list(sub_df[k])]).astype(np.float32)
    km = KMeans(
        n_clusters=len(np.unique(sub_df["lesion_id"].iloc[::])))
    sub_df["lesion_id"] = km.fit_predict(centers)
    return sub_df

def match_lesions_to_centers(sub_df,centers_class_df):
    k = "diagnostics_Mask-original_CenterOfMass"
    sub_ccdf = centers_class_df[
        centers_class_df["study_id"] == sub_df["study_id"].iloc[0]]
    sub_ccdf_centers = np.array(
        [x.strip().split(' ')
         for x in list(sub_ccdf["centers"])]).astype(np.float32)
    sub_df_centers = np.array(
        [x.split(':') for x in list(sub_df[k])]).astype(np.float32)
    km = KNeighborsClassifier(n_neighbors=1)
    sub_df["class"] = km.fit(
        sub_ccdf_centers,sub_ccdf["class"]).predict(sub_df_centers)
    return sub_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=desc)
    parser.add_argument(
        '--input_paths',dest='input_paths',type=str,nargs='+',required=True,
        help="Paths to sequences in nibabel compatible format.")
    parser.add_argument(
        "--id_pattern",dest="id_pattern",required=True,
        type=str,help="Pattern to extract ID from 'path' field in the data.")
    parser.add_argument(
        '--match_lesions',dest='match_lesions',action="store_true",
        help="Assumes lesion IDs are not correct and matches lesions between \
            modalities based on their relative distance to each other.")
    parser.add_argument(
        '--class_csv',dest='class_csv',action="store",default=None,type=str,
        help="Path to CSV with three columns: classes IDs, lesion centres \
            (spatial, format: x y z) and class. Classes will be converted to \
            numbers (after sorting their unique values) for compatibility.")
    parser.add_argument(
        '--path_patterns_to_sequence_id',dest='path_patterns_to_sequence_id',
        nargs='+',default=None,type=str,
        help="Uses regex patterns in the path column to redefine the \
            sequence ID (has to be in the format 'pattern1 pattern2').")
    args = parser.parse_args()

    output_dict = {}
    for path in args.input_paths:
        with open(path,'r') as o:
            data = json.load(o)
        if not np.all(np.isnan(data["lesion_id"][0])):
            # deletes lesions which were not considered because 
            # they were a single voxel
            skip_idxs = np.where(np.isnan(data["lesion_id"]))[0]
            for k in ['lesion_id','lesion_center','path','sequence_id','class']:
                data[k] = [data[k][i] for i,_ in enumerate(data[k])
                           if i not in skip_idxs]

            if 'registration' not in data:
                # compatibility with updated feature extraction script
                data["registration"] = ["rigid_translation" 
                                        for _ in data['lesion_id']]
            for k in data:
                v = data[k]
                v = [v[i] for i in range(len(v)) if i not in skip_idxs]
                if len(v) > 0:
                    if k in ["diagnostics_Mask-original_Size",
                            "diagnostics_Mask-original_CenterOfMass",
                            "diagnostics_Mask-original_CenterOfMassIndex"]:
                        v = [':'.join([str(y) for y in x]) for x in v]
                    if k == "sequence_id":
                        v = [int(x) for x in v]
                    if check_types(v[0],[int,float,str]):
                        if k in output_dict:
                            output_dict[k].extend(v)
                        else:
                            output_dict[k] = v
    
    output_dict["study_id"] = []
    for x in output_dict["path"]:
        j = re.search(args.id_pattern,x).group()
        output_dict["study_id"].append(j)

    output_df = pd.DataFrame.from_dict(output_dict)
    output_df["lesion_id"] = output_df["lesion_id"].astype(int)
    if args.match_lesions == True:
        output_df = output_df.groupby("study_id").apply(
            lambda x: match_lesions(x))

    if args.class_csv is not None:
        centers_class_df = pd.read_csv(args.class_csv)
        centers_class_df.rename(
            columns=dict(
                zip(centers_class_df.columns,["study_id","centers","class"])))
        class_correspondence = {
            x:i 
            for i,x in enumerate(
                np.sort(np.unique(centers_class_df["class"])))
        }
        centers_class_df["class"] = [
            class_correspondence[x] for x in centers_class_df["class"]]
        output_df = output_df.groupby("study_id").apply(
            lambda x: match_lesions_to_centers(x,centers_class_df))
    
    if args.path_patterns_to_sequence_id is not None:
        corr = {pat:i for i,pat in enumerate(
            args.path_patterns_to_sequence_id)}
        paths = output_df["path"].tolist()
        sequence_ids = [np.nan for _ in paths]
        for i,path in enumerate(paths):
            for k in corr:
                m = re.search(k,path)
                if m is not None:
                    sequence_ids[i] = corr[k]
        output_df["sequence_id"] = sequence_ids
    print(output_df.to_csv(index=False,index_label=None))