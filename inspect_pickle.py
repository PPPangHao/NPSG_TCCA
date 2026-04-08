import pickle, os, numpy as np, pprint
pfile = "/root/autodl-tmp/Code/rPPG-Toolbox/runs/exp/UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_TSCAN_UBFC-rPPG_outputs.pickle"
print("exists:", os.path.exists(pfile))
with open(pfile,"rb") as f:
    data = pickle.load(f)
print("type:", type(data))
if isinstance(data, dict):
    print("keys:")
    pprint.pprint(list(data.keys()))
    for k in list(data.keys()):
        v = data[k]
        try:
            arr = np.asarray(v)
            print(k, "-> shape", arr.shape, "dtype", arr.dtype)
        except Exception as e:
            print(k, "->", type(v), repr(v)[:200])
else:
    print("Data preview:", repr(data)[:1000])

