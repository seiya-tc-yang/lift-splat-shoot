import torch
import onnxruntime as ort
import numpy as np
from src.models import compile_model

def main():
    # config (same as original code)
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        'final_dim': (128, 352),
    }

    # torch model setup
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    modelf = "model525000.pt"
    model.load_state_dict(torch.load(modelf))
    model.eval()

    # onnx model set up
    onnx_session = ort.InferenceSession("model525000.onnx", providers=["CPUExecutionProvider"])


    # Test input (torch / onnx)
    # same shape as export_to_onnx.py
    B, N, C, H, W = 4, 6, 3, 128, 352
    x = torch.randn(B, N, C, H, W, dtype=torch.float32)
    rots = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    trans = torch.zeros(B, N, 3)
    intrins = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    post_rots = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1)
    post_trans = torch.zeros(B, N, 3)


    onnx_inputs = {
        onnx_session.get_inputs()[0].name: x.cpu().numpy(),
        onnx_session.get_inputs()[1].name: rots.cpu().numpy(),
        onnx_session.get_inputs()[2].name: trans.cpu().numpy(),
        onnx_session.get_inputs()[3].name: intrins.cpu().numpy(),
        onnx_session.get_inputs()[4].name: post_rots.cpu().numpy(),
        onnx_session.get_inputs()[5].name: post_trans.cpu().numpy(),
    }
    


    # torch inference
    with torch.no_grad():
        torch_out = model(x, rots, trans, intrins, post_rots, post_trans)
    
    torch_out_np = torch_out.cpu().numpy()


    # onnx inference
    onnx_out = onnx_session.run(None, onnx_inputs)[0]


    # comparison
    abs_diff = np.abs(torch_out_np - onnx_out)
    print("Max abs diff:", abs_diff.max())
    print("Mean abs diff:", abs_diff.mean())

    # close test
    if np.allclose(torch_out_np, onnx_out, atol=1e-4, rtol=1e-3):
        print("ONNX model output matches PT model")
    else:
        print("Outputs differ")


if __name__ == "__main__":
    main()




































