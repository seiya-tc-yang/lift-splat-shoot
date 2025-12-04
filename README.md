### Preparation

```
cd lift-splat-shoot
git switch modify_operator_for_onnx_conversion
python3.10 -m venv LSS_onnx_env
source LSS_onnx_env/bin/activate
pip3 install -r requirements.txt
```

### Convert to onnx model
Make sure you have pre-trained model model525000.pt under pwd
```
python export_to_onnx.py
```

### Evaluate onnx model with pt

```
python compare_onnx_pt.py
```

