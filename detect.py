import torch
import torchvision.transforms.functional as FT
import torchvision
from skimage import io
import albumentations as A
import sys
import os
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect(data_path, model):
    #change
    th = (200, 300, 1100, 900)
    transform_A = A.Compose([A.Crop(x_min=th[0], y_min=th[1], x_max=th[2], y_max=th[3], always_apply=True)])
    data_frame = pd.DataFrame(columns = ['image_name', 'x1', 'y1', 'x2', 'y2'])
    k = 0
    l = 0
    for img_nm in os.listdir(data_path):
        k+=1
        image = data_path + img_nm
        image = io.imread(image)
        image = transform_A(image=image)['image']
        #print(image)
        image_for_model = FT.to_tensor(image)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_for_model = FT.normalize(image_for_model, mean=mean, std=std)
        image_for_model = image_for_model.to(device)
        preds = model([image_for_model])[0]
        nms_preds = torchvision.ops.nms(preds['boxes'], preds['scores'], iou_threshold=0.2)
        result = {'boxes': preds['boxes'][nms_preds].to('cpu').tolist(),
                 'labels': preds['labels'][nms_preds].to('cpu').tolist()}
        for bbox, lbl in zip(result['boxes'], result['labels']):
            if lbl == 2:
                l+=1
                df = pd.Series({'image_name':img_nm, 'x1':bbox[0] + th[0], 'y1':bbox[1] + th[1],
                                   'x2':bbox[2] + th[0], 'y2':bbox[3] + th[1]})
                data_frame = data_frame.append(df, ignore_index = True)
    data_frame.to_csv('bboxes.csv')

if __name__ == '__main__':
    assert len(sys.argv) > 1
    data_path = sys.argv[1]
    checkpoint_path = 'checkpoint.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.eval()
    model.to(device)
    detect(data_path, model)