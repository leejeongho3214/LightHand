# Armo ✨✨
## Directory</br>
Build as the below architecture 
```
{$ROOT}
├── build
├── src
├── datasets
│   └─ freihand
│   └─ ArmoHand
│   └─ Etc.
└── models
    └─ hrnet
    └─ simplebaseline

```

## Models
```
cd {$ROOT}
cp -r /home/jeongho/tmp/armo_pose_estimation/models .
```

## datasets
Choose the dataset name you want
ex. name in freihand, ArmoHnad, etc.
```
cd {$ROOT}
mkdir datasets
cp -r /home/jeongho/tmp/Wearable_Pose_Model/datasets/{name} datasets/
```

## Train
```
cd {$ROOT}/src/tools
python model-name/dataset-name/name
ex. python hrnet/frei/2d
```

## args
you can change the epoch, count, init through arg command line
ex. python hrnet/frei/2d --epoch 100 --count 5 --reset

1. count means to stop the training when valid loss don't fall after series of 5 epoch

2. It saves automatically the check-point.pth whenever valid loss fall. \bar
Thus if you don't want to resume the check-point, insert "--reset"
