### Preparation
Download nuscenes data from [https://www.nuscenes.org/](https://www.nuscenes.org/).  
Set up environment and install dependencies.
```
cd lift-splat-shoot
git switch NOA-161_Separate_onnx_CAM_encode
python3.8 -m venv NOA-161_env
source NOA-161_env/bin/activate
pip3 install -r requirements.txt
```

### Separate and export onnx model
Separate LSS structure to CAM and BEV, export their respective class to onnx.

```
 python separative_export.py
```




