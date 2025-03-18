from unsloth import FastVisionModel
import torch
import json
from PIL import Image
from datasets import load_dataset

## Explore data preperation
dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
gt_data = json.loads(dataset[0]["ground_truth"])
with open('data.json', 'w') as f:
    json.dump(gt_data["gt_parse"], f)
image = dataset[0]["image"]
image.show()