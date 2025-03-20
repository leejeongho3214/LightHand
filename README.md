# LightHand

## Tool

### Randomize background OFF
<p align="left">
    <img src="assets/nobg.gif", width="640" height="480">
</p>

</br>

### Randomized background ON
<p align="left">
    <img src="assets/bg.gif", width="640" height="480">
</p>

### Download [[HERE](https://mlpa503.synology.me:15051/d/s/11e8WJFEehEJ8oO47O7LUuzq48naY9QX/6tU9h8kmIua_9586DLbZquVvTmS7gAi8-X78AAzEG9Qs)]

</br>

## Dataset
### Training set
<p align="left">
    <img src="assets/trainingset.png", style="width:850px;height:200px">
</p>

</br>

### Evaluation set
<p align="left">
    <img src="assets/evaluationset.png", style="width:850px;height:200px">
</p>

### Download [[HERE](https://mlpa503.synology.me:15051/d/s/11e8Wfd8LCZhQSxOlebul6ZF5dgUJuyj/jNhhV4wX7nsFcCEMthz8s7fH19lYSl0f-er5APyUF9Qs)]

## Directory</br>
Build as the below architecture 
```
{$ROOT}
├── build
├── src
├── datasets
│   └─ freihand
│   └─ LigthHand99K
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
python model_name/dataset_name/name
ex) python hrnet/frei/2d
```

## Args
you can change the epoch, count, init through arg command line
ex. python --name hrnet/frei/2d --epoch 100 --count 5 --reset

1. count means to stop the training when valid loss don't fall after series of 5 epoch

2. It saves automatically the check-point.pth whenever valid loss fall.
Thus if you don't want to resume the check-point, insert "--reset"
