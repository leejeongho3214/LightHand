import cv2
import mediapipe as mp
import os
import json
import numpy as np

import os
import sys
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.bar import colored
from torch.utils import data
from src.utils.argparser import load_model, pred_store, pred_test, parse_args
from src.tools.dataset import *
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For static images:
root = "../../datasets/test/rgb"
IMAGE_FILES = os.listdir(root)
with open(os.path.join(root, "../annotations.json"), "r") as f:
  ANNO_FILE = json.load(f)

total_image = len(IMAGE_FILES)
num = 0

def main():
  T_list = [['pckb', [0.1, 0.3]], ['mm', [0, 30]], ['mm', [0, 50]]]
  for T in T_list:
    t_list = thereshold_list(T[0], T[1])
    pred_test(T, t_list)
    
  print(f"Image ----> {num} / {total_image}")


def pred_test(T, t_list):
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(os.path.join(root, file)), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      gt_joint = np.array(ANNO_FILE[file.split('.')[0]]['coordinates'])[:, :-1] * 224
      
      # Print handedness and draw hand landmarks on the image.
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      joints = np.zeros((21, 2))
      for hand_landmarks in results.multi_hand_landmarks:
        for idx, joint in enumerate(hand_landmarks.landmark):
          joints[idx][0] = joint.x * 224
          joints[idx][1] = joint.y * 224
      
def thereshold_list(method, T_list):
    if method == "mm":
      thresholds_list = np.linspace(T_list[0], T_list[-1], 101)[1:] * 3.7795275591 ## change pixel coordinate to mm coordinate
    elif method == "pckb":
        thresholds_list = np.linspace(T_list[0], T_list[-1], 100)
    else: assert 0, "this method is the wrong"
    
    return thresholds_list

if __name__ == "__main__":
  main()