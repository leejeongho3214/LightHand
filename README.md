# Wearable_Pose_Model ✨✨


This is our research code of [Our_Model](docs/2022_KSCI.pdf). 

Our Model is a transformer-based for hand pose.</br>
The image is obtained from wrist-attached RGB camera.</br>
In this work, We study how to use our synthetic images instead of making real-world dataset.</br></br>

 <img src="docs/model.png" width="650"> </br></br>

## Directory</br>
Build as the below architecture 
```
{$ROOT}
|-- build
|-- src
|-- datasets
|-- models
|-- docs
```

## Setup with Conda</br>
```bash
git clone https://github.com/leejeongho3214/Wearable_Pose_Model.git
cd Wearable_Pose_Model
conda env create -f requirements.yaml
```
</br>


## Model Download</br>
Please download our files that need to run our code. [download](https://dkuniv-my.sharepoint.com/:f:/g/personal/72210297_dankook_ac_kr/Em6dacaP1AlNmTAmaBbX2osBxkTx8km8k7BeHT2d-TWF5A?e=poqt1A)</br>
</br>


## Train</br>
If you locates in Wearabel_Pose_Model folder, run the code below and the training will begin
```
cd src/tools
python train.py
```
</br>

## Solution</br>
If you get a error message such as the wrong path, follow the code below
```python
import sys
sys.path.append("/usr/your/path/Wearable_Pose_Model")
```
</br>

## Result</br>
### Evaluation Metric
* PCKb
    * The probablity of the correct keypoint within threshold
    * Threshold is according to hand bounding box 
* EPEv
    * The mean euclidean distance between ground truth and prediction of the visible joints (max 21)
* AUC under PCKb
    * Area under the curve (AUC)
    * It represents the percentage of correct keypoints (PCK) of which the Euclidean error is below a threshold t, according to hand bbox
* 3D MPJPE
    * Mean Per Joint Position Error (MPJPE)
    * Calculated after aligning the root joint (= wrist) of the estimated and groundtruth 3D pose </br>
### More details => [eval.md](docs/Evaluation.md)

### Paper => [paper.md](docs/paper.md)
