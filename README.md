# 🧪 <span>&#181;</span>ECT

Microscale 3-D Capacitence Tomography with a CMOS Sensor Array. 

## Download 3-D Dataset

Download sample 3-D dataset: 

```
./data/datasets/download.sh
```

The data will be downloaded to `data/datasets/3-D/dataset`

### Input and Output formats 

- Input is a 3-D matrix of capacitence measurements of size (m x n x r) = (20, 10, 5)
- Output is a 3-D volume of size (200, 100, 5) -> (200um, 100um, 50um)
  
## Training

```
  python3.7 train_3d.py --config <experiment-config> --exp_name <experiment-name>
```

For example, run the following to train on the downloaded dataset: 

```
  python3.7 train_3d.py --config config/experiments/3d/07112023.yaml --exp_name 07112023_3D > experiments/07112023_3D.log 2>&1 &
```

This by default outputs the logs to experiments/<experiment-name>

## Inference

```
  python3.7 evaluate_3d.py --config  <experiment-config> --model <path-to-model> --output_dir <output-dir>
```

For example, run the trained model: 

```
  python3.7 evaluate_3d.py --config  config/experiments/3d/07112023.yaml --model experiments/07112023_3D/best_model.pth 
```


## License 
BSD 3-Clause License. See [LICENSE](LICENSE). 
