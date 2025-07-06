import sys
sys.path.append('..')

import os
import cv2
import math
import torch
import torch.nn as nn
import joblib
import numpy as np
from PIL import Image,ImageFile
from torchvision import transforms
from utils.utils import FindContours
from sklearn.neighbors import KernelDensity
from torchvision.models import resnet50, ResNet50_Weights

ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None

NS, PS = 384, 2048
transforms_pp = transforms.Compose([
    transforms.Resize((NS, NS)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.638, 0.533, 0.575], [0.262, 0.280, 0.276])
])

transforms_if = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.638, 0.533, 0.575], [0.262, 0.280, 0.276])
])

X_plot = np.linspace(0,1,10)[:,np.newaxis]

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS" #设备+'--'+dpat[:-4]排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #设置第一块GPU可见
device = torch.device("cuda")
print(device)

model_glom_seg = torch.load('models/Glomerulus_Segmentation.pkl').to(device)
model_les_cat3 = torch.load('models/Glomerular_Lesion_Classification_Cate3.pkl').to(device)
model_les_cat2 = torch.load('models/Glomerular_Lesion_Classification_Cate2.pkl').to(device)

model_lf_score = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_lf_score.fc = nn.Sequential(nn.Linear(model_lf_score.fc.in_features, 224),
                           nn.ReLU(True),
                           nn.Linear(224, 28),
                           nn.ReLU(True),
                           nn.Linear(28,1))
model_lf_score.load_state_dict(torch.load('models/Immunofluorescence_Score.pth'))
model_lf_score = model_lf_score.to(device)

model_pat_cat4 = joblib.load('models/patient_classification_LMIF.pkl')

WI_path = 'examples/examples_lm_images/WSI.bmp'
IF_path = 'examples/examples_if_images/'

feature_org_c3_list = []
feature_org_c2_list = []

wsi_og = Image.open(WI_path).convert('RGB')
W,H = wsi_og.size
W_rs = math.ceil(W/PS)
H_rs = math.ceil(H/PS)

wsi_rs = wsi_og.resize((W_rs*PS,H_rs*PS), Image.BICUBIC)
wsi_rs_np = np.array(wsi_rs)
wsi_ex_np = np.zeros((H_rs*PS+PS,W_rs*PS+PS,3))
wsi_ex_np[PS//2:PS//2+H_rs*PS,PS//2:PS//2+W_rs*PS,:] = wsi_rs_np
wsi_ex = Image.fromarray(wsi_ex_np.astype('uint8')).convert('RGB')

prd_bi_np = np.zeros(((H_rs*NS+NS//2,W_rs*NS+NS//2)))

for i in range(W_rs*2+1):
    for j in range(H_rs*2+1):
        pch_og = wsi_ex.crop((i*PS//2,j*PS//2,i*PS//2+PS,j*PS//2+PS))
        pch_rs = pch_og.resize((NS,NS),Image.BICUBIC)
        pch_rs = transforms.ToTensor()(pch_rs)
        pch_rs = pch_rs.unsqueeze(0)
        prd_md = model_glom_seg(pch_rs)
        prd_bi = torch.round(prd_md).squeeze()
        prd_bi_ct = prd_bi[NS//4:NS//4*3,NS//4:NS//4*3].tolist()

        prd_bi_np[j*NS//2:(j+1)*NS//2,i*NS//2:(i+1)*NS//2] = prd_bi_ct


prd_bi_np = prd_bi_np*255
prd_bi = Image.fromarray(prd_bi_np[NS//4:H_rs*NS+NS//4,NS//4:W_rs*NS+NS//4].astype('uint8')).convert('1')
prd_bi = prd_bi.resize((W,H), Image.BICUBIC)

wsi_og = np.array(wsi_og)
prd_bi = np.array(prd_bi, dtype=np.uint8)*255

msk_contours = FindContours(prd_bi)
margin = 50
for contour in msk_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 1 and h > 1:
        cpglom = wsi_og[max(0, y - margin):min(y + h + margin, wsi_og.shape[0]),
                 max(0, x - margin):min(x + w + margin, wsi_og.shape[1]), :]
        cpglom = Image.fromarray(cpglom)

        cpglom = transforms_pp(cpglom).unsqueeze(0)
        feature_org_c3_list.append(torch.softmax(model_les_cat3(cpglom),dim=1).cpu().detach().numpy())
        feature_org_c2_list.append(torch.softmax(model_les_cat2(cpglom), dim=1).cpu().detach().numpy())

feature_org_c3_list = np.vstack(feature_org_c3_list)
feature_org_c2_list = np.vstack(feature_org_c2_list)

PAT_KDP_C3 = []
for i in range(feature_org_c3_list.shape[1]):
    X = feature_org_c3_list[:, i][:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
    PAT_KDP_C3 += np.exp(kde.score_samples(X_plot)).tolist()

PAT_KDP_C2 = []
for i in range(feature_org_c2_list.shape[1]):
    X = feature_org_c2_list[:, i][:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
    PAT_KDP_C2 += np.exp(kde.score_samples(X_plot)).tolist()

batch_if = []
for fname in os.listdir(IF_path):
    img = Image.open(os.path.join(IF_path, fname)).convert("RGB")
    img_tensor = transforms_if(img)
    batch_if.append(img_tensor)
batch_if = torch.stack(batch_if).to(device)
if_score = model_lf_score(batch_if)

feature_fnl_c4_list = np.array(PAT_KDP_C3 + PAT_KDP_C2 + if_score.flatten().tolist()).reshape(1,-1)
final_results = np.argmax(model_pat_cat4.predict_proba(feature_fnl_c4_list))

print('Diagnostic Result: {}'.format({0:'IgAN', 1:'MN', 2:'FSGS', 3:'MCD'}[final_results]))


