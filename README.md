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

</br>

## Dataset
### Training set
<p align="left">
    <img src="assets/trainingset.png", style="width:400px;height:200px">
</p>

</br>

### Evaluation set
<p align="left">
    <img src="assets/evaluationset.png", style="width:400px;height:200px">
</p>

### Download [[HERE](https://mlpa503.synology.me:15051/d/s/11e8C9aU6h8zlSqOdMqodkHMyvnRYddv/kuAILF61CCVg9DVR48jDnjfrxVQQ1HYr-DbPg8XIXKgw)]
### Above link consists both the dataset and the generator
### But, it need a password to access 
### please send an e-mail to 72210297@dankook.ac.kr, and I will provide you with pwd.

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
