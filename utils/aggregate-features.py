import json
import argparse
import numpy as np

desc = """
Aggregates multiple json files and prints them in a comma separated format.
"""

def check_types(x,types):
    for t in types:
        if isinstance(x,t) == True:
            return True
    return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=desc)
    parser.add_argument(
        '--input_paths',dest='input_paths',type=str,nargs='+',required=True,
        help="Paths to sequences in nibabel compatible format.")
    args = parser.parse_args()

    output_dict = {}
    for path in args.input_paths:
        with open(path,'r') as o:
            data = json.load(o)
        if 'lesion_id' not in data:
            print(path)
        if not np.isnan(data["lesion_id"][0]):
            if 'registration' not in data:
                # compatibility with updated feature extraction script
                data["registration"] = ["rigid_translation" 
                                        for _ in data['lesion_id']]
            for k in data:
                v = data[k]
                if check_types(v[0],[int,float,str]):
                    if k in output_dict:
                        output_dict[k].extend(v)
                    else:
                        output_dict[k] = v
    
    n = len(output_dict["lesion_id"])
    # print a header
    p = []
    for k in output_dict:
        p.append(str(k))
    print(",".join(p))
    for i in range(n):
        p = []
        for k in output_dict:
            p.append(str(output_dict[k][i]))
        print(",".join(p))