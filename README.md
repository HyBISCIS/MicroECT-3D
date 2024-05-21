# 🧪 <span>&#181;</span>ECT

Microscale 3-D Capacitence Tomography with a CMOS Sensor Array. 

## Download 3-D Dataset

Download sample 3-D dataset: 

```
./download.sh
```

The data will be downloaded to `data/datasets/3-D/dataset`

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
