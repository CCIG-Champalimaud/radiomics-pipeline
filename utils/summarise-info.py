import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Returns the quantile of a field for a JSON")

    parser.add_argument('--input_path',dest='input_path')
    parser.add_argument('--parameter',dest='parameter',default="spacing")
    parser.add_argument('--quantile',dest='quantile',default=0.5,type=float)
    args = parser.parse_args()

    with open(args.input_path) as o:
        data_dict = json.load(o)

    all_info = np.array(
        [data_dict[k][args.parameter] for k in data_dict])
    print(",".join(
        [str(x) 
         for x in np.quantile(all_info,args.quantile,axis=0).tolist()]))