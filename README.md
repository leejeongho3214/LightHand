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
You can download these models through the below link.


## Datasets
You can download these models through the below link.
It consist of ArmoHAND, FreiHAND and RHD dataset


## Train
```
cd {$ROOT}/src/tools
python model-name/dataset-name/name
ex) python hrnet/frei/2d
```

## Args
you can change the epoch, count, init through arg command line
ex. python hrnet/frei/2d --epoch 100 --count 5 --reset

1. count means to stop the training when valid loss don't fall after series of 5 epoch

2. It saves automatically the check-point.pth whenever valid loss fall. \bar
Thus if you don't want to resume the check-point, insert "--reset"
