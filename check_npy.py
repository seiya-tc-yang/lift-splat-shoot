import numpy as np

def main():
    lss = np.load("pt.npy")
    onnx = np.load("onnx.npy")

    # Shape Check
    print("-----Shape Check-----")
    print("pt shape: ",lss.shape)
    print("onnx shape: ",onnx.shape)
    print("Same shape: ", lss.shape == onnx.shape)
    print("")

    # Value Check
    print("-----Value Check-----")
    diff = lss - onnx
    print("Max diff: ", np.max(diff))
    print("Mean diff: ", np.mean(diff))
    idx = np.unravel_index(np.argmax(diff), lss.shape)
    print("Max diff index:", idx)
    print("Orig value:", lss[idx], "ONNX value:", onnx[idx])
    print("")

    # Close Test
    print("-----Close Test-----")
    print("np.allclose: ", np.allclose(lss, onnx, atol=1e-5, rtol=1e-3))
    print("")

    # Top-k Max Diffs
    k = 5 
    flat_diff = diff.flatten()
    topk_indices = np.argpartition(flat_diff, -k)[-k:]
    topk_indices = topk_indices[np.argsort(-flat_diff[topk_indices])]  # order (big to small)

    print(f"-----Top {k} Max Differences-----")
    for i, flat_idx in enumerate(topk_indices):
        idx = np.unravel_index(flat_idx, diff.shape)
        print(f"NO. {i+1}: diff={diff[idx]:.6f}  index={idx}  "
              f"PT={lss[idx]:.6f}  ONNX={onnx[idx]:.6f}")
  

if __name__ == "__main__":
    main()






