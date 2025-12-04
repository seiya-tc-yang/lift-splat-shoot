import torch
import onnxruntime as ort
import numpy as np
from src.models import compile_model

def main():
    # config
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        'final_dim': (128, 352),
    }
   
    # model setup
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    modelf = "model525000.pt"
    model.load_state_dict(torch.load(modelf))
    model.eval()
   
    # deny quickcumsum
    for module in model.modules():
        if hasattr(module, 'use_quickcumsum'):
            module.use_quickcumsum = False
            print(f"Set {module.__class__.__name__}.use_quickcumsum = False")
   
    # dummy inputs
    B, N, C, H, W = 4, 1, 3, 128, 352
    imgs = torch.randn(B, N, C, H, W, dtype=torch.float32)
    rots = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    trans = torch.zeros(B, N, 3)
    intrins = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    post_rots = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    post_trans = torch.zeros(B, N, 3)
    

    device = next(model.parameters()).device
    print("device:", device)
    dummy_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
   
    # test PyTorch
    print("Testing PyTorch model...")
    with torch.no_grad():
        pt_output = model(*dummy_inputs)
        print(f"PyTorch output shape: {pt_output.shape}")
        print(f"PyTorch output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")
   
    
    # export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_inputs,
        "model525000.onnx",
        export_params=True,
        opset_version=17,  
        do_constant_folding=True,
        input_names=["imgs", "rots", "trans", "intrins", "post_rots", "post_trans"],
        output_names=["bev_feature"],
        dynamic_axes={
            "imgs": {0: "batch", 1: "num_cams"},
            "rots": {0: "batch", 1: "num_cams"},
            "trans": {0: "batch", 1: "num_cams"},
            "intrins": {0: "batch", 1: "num_cams"},
            "post_rots": {0: "batch", 1: "num_cams"},
            "post_trans": {0: "batch", 1: "num_cams"},
            "bev_feature": {0: "batch"},
        },
        dynamo=False,  # use old TorchScript-based to export
    )
    print("Exported to ONNX successfully")
    # =========================================
   
    # test ONNX
    print("\nEvaluating ONNX model...")
   
    ort_session = ort.InferenceSession("model525000.onnx", providers=['CPUExecutionProvider'])
   
    onnx_inputs = {
        'imgs': imgs.cpu().numpy(),
        'rots': rots.cpu().numpy(),
        'trans': trans.cpu().numpy(),
        'intrins': intrins.cpu().numpy(),
        'post_rots': post_rots.cpu().numpy(),
        'post_trans': post_trans.cpu().numpy(),
    }
   
    onnx_output = ort_session.run(None, onnx_inputs)[0]
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
   
    # 比較
    diff = np.abs(pt_output.numpy() - onnx_output)
    print(f"\nDifference Statistics:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
   
    if np.allclose(pt_output.numpy(), onnx_output, atol=1e-4, rtol=1e-3):
        print("\n(O) ONNX export successful and outputs match PyTorch.")
    else:
        print("\n(X) ONNX output differs from PyTorch")

if __name__ == "__main__":
    main()

