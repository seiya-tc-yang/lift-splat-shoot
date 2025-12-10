import onnx
import onnxruntime as ort
import torch
import torch.onnx
import numpy as np
import traceback
from src.models import compile_model

def export_CAM_encode(model, output_path="CAM_encode.onnx"):    
    print("\n========== CAM encode ==========")

    # Extract CAM Encode 
    cam_encode = model.camencode
    cam_encode.eval()
   
    batch_size = 4
    n_cams = 6
    channels = 3
    H, W = 128, 352  # final_dim from data_aug_conf
    
    dummy_input = torch.randn(batch_size, n_cams, channels, H, W)
    print(f"Img original input shape: {dummy_input.shape}")  # should be [4, 6, 3, 128, 352]

    dummy_input = dummy_input.view(batch_size*n_cams, channels, H, W)    
    print(f"CAM encode input shape ( after B*N ): {dummy_input.shape}")  # should be [24, 3, 128, 352]
    
    # test forward
    with torch.no_grad():
        output = cam_encode(dummy_input)
        print(f"PyTorch model output shape: {output.shape}")  # should be [24, 64, 41, 8, 22]
    

    print("\nExporting CAM encode......")
    torch.onnx.export(
        cam_encode,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['imgs'],  
        output_names=['features'],  
        dynamic_axes={
            'imgs': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"\nCAM encode converted to: {output_path}")
    
    return output_path

def export_BEV_encode(model, output_path="BEV_encode.onnx"):
    print("\n========== BEV encode ==========")

    # Extract BEV Encode 
    bev_encoder = model.bevencode
    bev_encoder.eval()
    
    batch_size = 4
    Conv = 64 
    X = 200
    Y = 200
    
    dummy_input = torch.randn(batch_size, Conv, X, Y)
    
    print(f"BEV encode input shape: {dummy_input.shape}")  # should be [4, 64, 200, 200]
    
    # test forward
    with torch.no_grad():
        output = bev_encoder(dummy_input)
        print(f"PyTorch model output shape: {output.shape}")  # should be [4, 1, 200, 200]
    

    print("\nExporting BEV encode......")
    torch.onnx.export(
        bev_encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['bev_features'],
        output_names=['output'],
        dynamic_axes={
            'bev_features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"BEV encode converted to: {output_path}")
    
    return output_path


def verify_onnx_model(onnx_path, dummy_input):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX structure check pass")

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX inference succeeded")
    print(f"ONNX model output shape: {ort_outputs[0].shape}")
    
    return ort_outputs



if __name__ == "__main__":
    model_path = "model525000.pt"

    # for LSS's original compile model (models.py)
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        'resize_lim': (0.193, 0.225),
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
    }

    
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # Test CAM
    try:
        cam_onnx_path = export_CAM_encode(model)
        
        dummy_input = torch.randn(24, 3, 128, 352)  # ingore magic num for now
        verify_onnx_model(cam_onnx_path, dummy_input)
        
    except Exception as e:
        print(f"CAM encode separative conversion failed: {e}")
        
        traceback.print_exc()
    
    
    # Test BEV
    try:
        bev_encoder_path = export_BEV_encode(model)
        
        dummy_input = torch.randn(4, 64, 200, 200)  # ingore magic num for now
        verify_onnx_model(bev_encoder_path, dummy_input)
        
    except Exception as e:
        print(f"BEV encode separative conversion failed: {e}")
        traceback.print_exc()