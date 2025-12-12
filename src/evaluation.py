import torch
from src import models
from src import data
import onnxruntime as ort
import numpy as np
import os
import sys

def output_value_test(out_pt, out_onnx):

    # Shape Check
    print("-----Shape Check-----")
    print("pt shape: ",out_pt.shape)
    print("onnx shape: ",out_onnx.shape)
    print("Same shape: ", out_pt.shape == out_onnx.shape)

    # Value Check
    print("\n-----Value Check-----")
    diff = out_pt - out_onnx
    print("Max diff: ", np.max(diff))
    print("Mean diff: ", np.mean(diff))
    idx = np.unravel_index(np.argmax(diff), out_pt.shape)
    print("Max diff index:", idx)
    print("PT value:", out_pt[idx], "ONNX value:", out_onnx[idx])
    print(f"Max relative error: {((np.max(diff)/out_pt[idx])*100):.3f} %")

    # Close Test
    print("\n-----Numpy Close Test-----")
    print("np.allclose: ", np.allclose(out_pt, out_onnx, atol=0.01, rtol=0.01))
    # Stricter TH
    #print("np.allclose: ", np.allclose(out_pt, out_onnx, atol=1e-2, rtol=1e-3))
     

    # Top-k Max Diffs
    k = 5 
    flat_diff = diff.flatten()
    topk_indices = np.argpartition(flat_diff, -k)[-k:]
    topk_indices = topk_indices[np.argsort(-flat_diff[topk_indices])]  # order (big to small)
    print(f"\n-----Top {k} Max Differences-----")
    for i, flat_idx in enumerate(topk_indices):
        idx = np.unravel_index(flat_idx, diff.shape)
        print(f"NO. {i+1}: diff={diff[idx]:.6f}  index={idx}  "
              f"PT={out_pt[idx]:.6f}  ONNX={out_onnx[idx]:.6f}")


def compare_models( PTmodelf,
                    ONNXmodelf,
                    section='CAM',
                    version='mini',
                    dataroot='data',
                    gpuid=0,
                    viz_train=False,
                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],
                    bsz=4,
                    nworkers=10,
                    ):
    

    # To avoid BEV for now
    if section != 'CAM' or 'bev' in ONNXmodelf.lower():
        print("BEV model evaluation not developed yet.")
        sys.exit()

    
    # config
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 6,
                }
    

    # Data & Device preparation
    trainloader, valloader = data.compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')


    # PyTorch Model Inference
    print('\nStarting PyTorch inference...')
    model = models.compile_model(grid_conf, data_aug_conf, outC=1)
    model.load_state_dict(torch.load(PTmodelf))
    model.eval()
    cam_encode = model.camencode.to(device)

    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            B, N, C, imH, imW = imgs.shape
            imgs = imgs.view(B*N, C, imH, imW)
            print("PT model input: ", imgs.shape)
            
            out_pt = cam_encode(imgs.to(device))
            print("PT model output: ", out_pt.shape)
            out_pt = out_pt.detach().cpu().numpy()

            # we need only one iteration for correct CAM encode input shape
            break

    
    # ONNX Model Inference
    print('\nStarting ONNX inference...')
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"Using CUDA provider on GPU")
    else:
        providers = ['CPUExecutionProvider']
        print("CUDAExecutionProvider not available, use CPU instead.")

    ort_session_CAM = ort.InferenceSession(ONNXmodelf, providers=providers)
    imgs = imgs.detach().cpu().numpy()
    print("ONNX model input: ", imgs.shape)
    out_onnx = (ort_session_CAM.run(None, {'imgs': imgs}))[0]
    print("ONNX model output: ", out_onnx.shape)


    # Value test & evaluation
    print('\nStarting output comparison...')
    output_value_test(out_pt, out_onnx)

    


