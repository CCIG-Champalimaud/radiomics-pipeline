import argparse
import json
import time
import numpy as np
import itk
import SimpleITK as sitk
from skimage import measure
from radiomics import featureextractor
from radiomics import setVerbosity

setVerbosity(40)

def print_verbose(*string,verbose):
    if verbose == True:
        print(' '.join(string))

def correct_bias_field(image,n_fitting_levels,n_iter,shrink_factor=1):
    image_ = image
    if shrink_factor > 1:
        image_ = sitk.Shrink(
            image_,[shrink_factor]*image_.GetDimension())
    mask_image = sitk.OtsuThreshold(image_)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_fitting_levels * [n_iter])
    log_bf = corrector.GetLogBiasFieldAsImage(image)
    corrected_input_image = image/sitk.Exp(log_bf)
    return corrected_input_image

def normalize_values(sitk_image):
    arr = sitk.GetArrayFromImage(sitk_image)
    m,M = arr.min(),arr.max()
    sitk_image = sitk.Subtract(sitk_image,float(m))
    sitk_image = sitk.Divide(sitk_image,float(M-m))
    return sitk_image

def pad_to_size(sitk_image,target_image):
    sh1 = np.array(sitk_image.GetSize(),dtype=np.uint32)
    sh2 = np.array(target_image.GetSize(),dtype=np.uint32)
    diff = sh2-sh1
    lower = np.uint32(np.floor(diff//2))
    upper = diff - lower
    return sitk.ConstantPad(sitk_image,lower.tolist(),upper.tolist(),0.0)

def resample_label(sitk_image,out_spacing=[1.0, 1.0, 1.0],
                   thr=0.2,target=None):
    # works better than NN sampling for masks with small annotated regions
    if target is None:
        target = sitk_image
    original_spacing = sitk_image.GetSpacing()
    if original_spacing != out_spacing:
        original_size = sitk_image.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetDefaultPixelValue(0.0)
        resample.SetSize(out_size)
        resample.SetOutputDirection(target.GetDirection())
        resample.SetOutputOrigin(target.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0.0)
        resample.SetInterpolator(sitk.sitkLinear)

        output = sitk.Image(out_size,sitk.sitkUInt8)
        output.SetOrigin(target.GetOrigin())
        output.SetDirection(target.GetDirection())
        output.SetSpacing(out_spacing)
        binarizer = sitk.BinaryThresholdImageFilter()
        binarizer.SetLowerThreshold(thr)
        for lesion_mask,_,cl in yield_lesion_masks(sitk_image):
            lesion_mask = sitk.Cast(lesion_mask,sitk.sitkFloat32)
            binarizer.SetInsideValue(int(cl))
            resampled_image = resample.Execute(lesion_mask)
            binarized_image = binarizer.Execute(resampled_image)
            binarized_image.SetOrigin(target.GetOrigin())
            binarized_image.SetDirection(target.GetDirection())
            output = sitk.Add(output,binarized_image)
        output = sitk.Cast(output,sitk.sitkFloat32)
        return output

    else:
        return sitk_image

def register_label(itk_image,fixed,parameters,thr=0.2):
    # works better than NN sampling for masks with small annotated regions
    sitk_image = itk_to_sitk(itk_image)
    fixed = itk_to_sitk(fixed)
    output = sitk.Image(fixed.GetSize(),sitk.sitkUInt8)
    output.CopyInformation(fixed)
    binarizer = sitk.BinaryThresholdImageFilter()
    binarizer.SetLowerThreshold(thr)
    for lesion_mask,_,cl in yield_lesion_masks(sitk_image):
        lesion_mask = sitk.Cast(lesion_mask,sitk.sitkFloat32)
        binarizer.SetInsideValue(int(cl))
        lesion_mask = sitk_to_itk(lesion_mask)
        registered_image = itk.transformix_filter(lesion_mask,parameters)
        registered_image = itk_to_sitk(registered_image)
        binarized_image = binarizer.Execute(registered_image)
        output = sitk.Add(output,binarized_image)
    output = sitk.Cast(output,sitk.sitkFloat32)
    output = sitk_to_itk(output)
    return output

def resample_image(sitk_image,out_spacing=[1.0, 1.0, 1.0],is_label=False):
    original_spacing = sitk_image.GetSpacing()
    if original_spacing != out_spacing:
        original_size = sitk_image.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetDefaultPixelValue(0.0)
        resample.SetSize(out_size)
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0.0)

        if is_label == True:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(sitk_image)

    else:
        return sitk_image

def resample_image_to_target(moving,target,is_label=False):
    if is_label == True:
        interpolation = sitk.sitkNearestNeighbor
    else:
        interpolation = sitk.sitkBSplineResampler
    output = sitk.Resample(
        moving,target.GetSize(),sitk.Transform(), 
        interpolation,
        target.GetOrigin(),target.GetSpacing(),target.GetDirection(),
        0,moving.GetPixelID())
    return output

def sitk_to_itk(sitk_image):
    # source: https://discourse.itk.org/t/in-python-how-to-convert-between-simpleitk-and-itk-images/1922
    image_dimension = sitk_image.GetDimension()
    itk_image = itk.GetImageFromArray(
        sitk.GetArrayFromImage(sitk_image),
        is_vector = sitk_image.GetNumberOfComponentsPerPixel()>1)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())   
    itk_image.SetDirection(
        itk.GetMatrixFromArray(
            np.reshape(
                np.array(sitk_image.GetDirection()), [image_dimension]*2)))
    return itk_image

def itk_to_sitk(itk_image):
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image),
        isVector=itk_image.GetNumberOfComponentsPerPixel()>1)
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    sitk_image.SetDirection(
        itk.GetArrayFromMatrix(
            itk_image.GetDirection()).flatten())
    return sitk_image

def yield_lesion_masks(mask):
    mask_arr = sitk.GetArrayFromImage(mask)
    blobs = measure.label(mask_arr,background=0)
    for i in np.unique(blobs):
        if i != 0:
            lesion_arr = np.zeros_like(blobs)
            x,y,z = np.where(blobs == i)
            cl = np.median(mask_arr[x,y,z])
            lesion_arr[x,y,z] = 1
            center = [
                np.mean([x.max(),x.min()]),
                np.mean([y.max(),y.min()]),
                np.mean([z.max(),z.min()])]
            lesion_mask = sitk.GetImageFromArray(lesion_arr)
            lesion_mask.CopyInformation(mask)
            yield lesion_mask,center,cl

desc = """
Extracts features for a given set of sequences and a single mask.
In case the sequences or the mask are of different sizes, it maps 
all of the sequences to a common space given as input. 

Optionally, this script also registers all sequences assuming that the
first sequence is the reference. The mask may also be registered if its
input size is different from that of the first input sequence.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=desc)
    parser.add_argument(
        '--input_paths',dest='input_paths',type=str,nargs='+',required=True,
        help="Paths to sequences in nibabel compatible format.")
    parser.add_argument(
        '--configs',dest='configs',type=str,nargs='+',required=True,
        help="Paths to pyradiomics configuration files.")
    parser.add_argument(
        '--mask_path',dest='mask_path',type=str,required=True,
        help="Path to mask in nibabel compatible format.")
    parser.add_argument(
        '--target_spacing',dest='target_spacing',type=float,default=None,
        nargs="+",help="Target spacing for the inputs/mask.")
    parser.add_argument(
        '--output_path',dest='output_path',type=str,
        required=True,help="Output path.")
    parser.add_argument(
        '--registration',dest='registration',type=str,default="mask",
        choices=["mask","largest","first","none"],
        help="Registers all images to an image with the shape of the mask \
            or to the largest image (registration is inferred from the first \
            non-fixed image and applied to other images).")
    parser.add_argument(
        '--registration_parameter_files',dest='registration_parameter_files',
        nargs="+",help="Path to ilastix registration parameters.")
    parser.add_argument(
        '--assume_same',dest="assume_same",action="store_true",default=False,
        help="Assumes that if fixed and moving images/masks have the same shape \
            they are already co-registered (or equivalent). Only for \
            registration == 'first'.")
    parser.add_argument(
        '--verbose',dest="verbose",action="store_true",default=False,
        help="Activates verbosity.")
    parser.add_argument(
        '--cond_mult_idx',dest='cond_mult_idx',type=int,default=None,
        help="Image path index to apply conditional multiplication.")
    parser.add_argument(
        '--conditional_multiplication',dest='conditional_multiplication',
        type=float,nargs=2,
        help="Two values, [0] sets a threshold and if any value in the image \
            is above that threshold, then the whole image is multiplied by \
            [1].")
    parser.add_argument(
        '--minimum_lesion_size',dest="minimum_lesion_size",default=2,type=float,
        help="Defines the minimum lesion size in voxels")

    args = parser.parse_args()

    time_a = time.time()

    sf = sitk.sitkFloat32
    input_dict = {i:k for i,k in enumerate(args.input_paths)}
    args.configs = {i:k for i,k in enumerate(args.configs)}
    all_sequences = {k:sitk.ReadImage(input_dict[k],sf) for k in input_dict}
    mask = sitk.ReadImage(args.mask_path,sf)
    
    unique_labels = np.unique(sitk.GetArrayFromImage(mask))
    if len(unique_labels) == 1:
        print_verbose(
            "No lesions in mask!",verbose=args.verbose)
        output_dict = {
            "path":args.input_paths,
            "lesion_center":[np.nan for _ in args.input_paths],
            "lesion_id":[np.nan for _ in args.input_paths],
            "sequence_id":[i for i in range(len(args.input_paths))],
            "class":[np.nan for _ in args.input_paths],
            "config_file":[np.nan for _ in args.configs],
            "mask_resampling":[np.nan for _ in args.configs],
            "registration":[np.nan for _ in args.configs]}
        output = json.dumps(output_dict,indent=2)
        with open(args.output_path,'w') as o:
            o.write(output)
        exit()

    # resample images and masks
    all_sequences = {
        k:resample_image(all_sequences[k],args.target_spacing) 
        for k in all_sequences}
    mask_ = resample_image(mask,args.target_spacing,is_label=True)

    if len(np.unique(sitk.GetArrayFromImage(mask_))) != len(unique_labels):
        # if NN sampling accidentally eliminates labels (can happen when 
        # objects are very small) we resample 
        print_verbose("Repeating mask resampling to avoid losing lesions",
                      verbose=args.verbose)
        mask_ = resample_label(mask,args.target_spacing,thr=0.35)
        mask_resampling = "linear"
    else:
        mask_resampling = "nearest_neighbour"
    mask = mask_

    # assign mask to its respective sequence (if any) and ensure that the
    # spacings, directions and origins are identical
    all_sh = {k:all_sequences[k].GetSize() for k in all_sequences}
    mask_sh = mask.GetSize()
    same_size_as_mask = []
    for k in all_sh:
        if all_sh[k] == mask_sh:
            same_size_as_mask.append(k)
    if len(same_size_as_mask) > 0:
        c = same_size_as_mask[0]
        mask.SetSpacing(all_sequences[c].GetSpacing())
        mask.SetDirection(all_sequences[c].GetDirection())
        mask.SetOrigin(all_sequences[c].GetOrigin())

    # register images
    rpf = args.registration_parameter_files
    if args.registration != 'none' and rpf is not None:
        # aligning using translation and rigid-body registration
        parameter_object = itk.ParameterObject.New()
        for file in rpf:
            parameter_object.AddParameterFile(file)
        
    all_sequences = {k:sitk_to_itk(all_sequences[k]) for k in all_sequences}
    mask = sitk_to_itk(mask)

    reg_mask = True
    if args.registration == "mask":
        print_verbose(
            "Registering all images/mask to the one corresponding to the mask",
            verbose=args.verbose)
        c = same_size_as_mask[0]
        no_reg = same_size_as_mask

    elif args.registration == "largest":
        print_verbose(
            "Registering all images/mask to the largest image",
            verbose=args.verbose)
        no_reg = np.argmax(
            [np.prod(all_sequences[k].shape) for k in all_sequences])
        try:
            c = no_reg[0]
        except:
            no_reg = [no_reg]
            c = no_reg[0]
    
    elif args.registration == "first":
        print_verbose(
            "Registering all images/mask to the first image",
            verbose=args.verbose)
        no_reg = [0]
        c = no_reg[0]
        if args.assume_same == True:
            for k in range(1,len(all_sh)):
                sh = all_sh[k]
                if np.all(sh == all_sh[0]):
                    no_reg.append(k)

    elif args.registration == "none":
        print_verbose(
            "No registration, only resampling",
            verbose=args.verbose)
        no_reg = [i for i in all_sequences]
        c = 0
        fixed = all_sequences[c]
        fixed_sitk = itk_to_sitk(fixed)
        for k in all_sequences:
            all_sequences[k] = itk_to_sitk(all_sequences[k])
            all_sequences[k] = resample_image_to_target(
                all_sequences[k],fixed_sitk)
        mask = resample_image_to_target(itk_to_sitk(mask),fixed_sitk,True)
        reg_out = ["none"]

    reg = sorted([i for i in all_sequences if i not in no_reg])

    fixed = all_sequences[c]
    moving = {k:all_sequences[k] for k in reg}
        
    if len(moving) > 0:
        fixed_sitk = itk_to_sitk(fixed)
        for k in moving:
            moving[k] = itk_to_sitk(moving[k])
            moving[k] = resample_image_to_target(moving[k],fixed_sitk)
            moving[k] = sitk_to_itk(moving[k])
        mask = sitk_to_itk(
            resample_image_to_target(itk_to_sitk(mask),fixed_sitk,True))

        stop_reg = False
        reg_attempts = 0
        reg_out = ["translation","rigid"]
        while stop_reg == False:
            m = moving[reg[0]]
            moving_params = {
                "spacing":m.GetSpacing(),
                "origin":m.GetOrigin(),
                "direction":m.GetDirection()}
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed,m,parameter_object=parameter_object,
                log_to_console=False)
            result_images = {reg[0]:itk_to_sitk(result_image)}
            if args.registration != "mask" and c not in same_size_as_mask:
                # change to nearest neighbour for mask
                result_transform_parameters.SetParameter(
                    "Interpolator",
                    "NearestNeighborInterpolator")
                result_transform_parameters.SetParameter(
                    "ResampleInterpolator",
                    "FinalNearestNeighborInterpolator")
                result_transform_parameters.SetParameter(
                    "FinalBSplineInterpolationOrder","0")

                mask_ = itk.transformix_filter(
                    mask,result_transform_parameters)
                N = len(np.unique(itk.GetArrayFromImage(mask_)))
                if N != len(unique_labels):
                    result_transform_parameters.SetParameter(
                        "Interpolator","BSplineInterpolator")
                    result_transform_parameters.SetParameter(
                        "ResampleInterpolator","FinalBSplineInterpolator")
                    result_transform_parameters.SetParameter(
                        "FinalBSplineInterpolationOrder","1")
                    mask_ = register_label(
                        mask,fixed,result_transform_parameters,0.2)
                mask = mask_
            N = len(np.unique(itk.GetArrayFromImage(mask)))
            if N == len(unique_labels):
                stop_reg = True
            else:
                print_verbose(
                    "Translation+rigid body registration failed, attempting only translation registration...",
                    verbose=args.verbose)
                parameter_object = itk.ParameterObject.New()
                parameter_object.AddParameterFile(
                    "registration-parameters/translation.txt")
                reg_out = ["translation"]

            if reg_attempts > 1:
                reg_out = []
                stop_reg = True
            
            reg_attempts += 1
        
        mask = itk_to_sitk(mask)
        
        for k in reg[1:]:
            mi = moving[k]
            transf_im = itk.transformix_filter(
                mi,result_transform_parameters)
            transf_im = itk_to_sitk(transf_im)
            result_images[k] = transf_im
        
        out = result_images
        for k in sorted(no_reg):
            out[k] = itk_to_sitk(all_sequences[k])
        all_sequences = out

    elif args.registration != "none":
        mask = itk_to_sitk(mask)
        for k in all_sequences:
            all_sequences[k] = itk_to_sitk(all_sequences[k])
        # in practical terms no registration occurred
        reg_out = ["none"]

    reg_out_str = '_'.join(reg_out) if len(reg_out)>0 else "failed"
    if reg_out_str == "failed":
        print_verbose(
            "\tRegistration failed, skipping feature extraction",
            verbose=args.verbose)
        features = {}
        output_dict = {
            "path":args.input_paths,
            "lesion_center":[np.nan for _ in args.input_paths],
            "lesion_id":[np.nan for _ in args.input_paths],
            "sequence_id":[i for i in range(len(args.input_paths))],
            "class":[np.nan for _ in args.input_paths],
            "config_file":[np.nan for _ in args.configs],
            "mask_resampling":[np.nan for _ in args.configs],
            "registration":["failed" for _ in args.configs]}
        output = json.dumps(output_dict,indent=2)
        with open(args.output_path,'w') as o:
            o.write(output)
        exit()

    output_dict = {}
    for k in all_sequences:
        config = args.configs[k]
        sequence = all_sequences[k]
        if k == args.cond_mult_idx:
            am = args.conditional_multiplication
            M = sitk.GetArrayFromImage(sequence).max()
            if M > am[0]:
                sequence = sitk.Multiply(
                    sitk.Cast(sequence,sitk.sitkFloat32),am[1])
        fe = featureextractor.RadiomicsFeatureExtractor(config)
        path = input_dict[k]
        for i,(lesion_mask,center,cl) in enumerate(yield_lesion_masks(mask)):
            print_verbose(
                "Extracting features for lesion {} in {}".format(i,path),
                verbose=args.verbose)
            print_verbose(
                "\tConfig file: "+config,
                verbose=args.verbose)
            print_verbose(
                "\tCenter: {}\n\tClass: {}".format(center,cl),
                verbose=args.verbose)
            time_c = time.time()
            mls = args.minimum_lesion_size
            if np.count_nonzero(sitk.GetArrayFromImage(lesion_mask)) < mls:
                print_verbose(
                    "\tOnly one voxel present in lesion, skipping",
                    verbose=args.verbose)
                features = {}
                features["lesion_id"] = np.nan
                features["lesion_center"] = center
                features["path"] = path
                features["sequence_id"] = k
                features["class"] = cl
            else:
                features = fe.execute(sequence,lesion_mask)
                features = dict(features)
                features["lesion_id"] = i
                features["path"] = path
                features["lesion_center"] = center
                features["sequence_id"] = k
                features["class"] = cl
                features["config_file"] = config
                features["registration"] = reg_out_str
                features["mask_resampling"] = mask_resampling
            for key in features:
                value = features[key]
                try: value = float(value)
                except: pass
                if key in output_dict:
                    output_dict[key].append(value)
                else:
                    output_dict[key] = [value]
            time_d = time.time()
            print_verbose(
                "\tTime elapsed in this lesion: {}".format(time_d-time_c),
                verbose=args.verbose)

    output = json.dumps(output_dict,indent=2)
    with open(args.output_path,'w') as o:
        o.write(output)
    
    time_b = time.time()
    print_verbose("Done!",verbose=args.verbose)
    print_verbose(
        "Total time elapsed: {}".format(time_b-time_a),
        verbose=args.verbose)