#!/usr/bin/env python
# coding: utf-8

# # Leave One Subject Out CV

# - 기존에는 전체 데이터를 10-fold로 나누고 하나는 테스트, 하나는 검정, 나머지는 훈련에 사용
# 
# - 수정된 버전의 경우 전체 데이터에서 테스트 데이터를 먼저 분리한 후 나머지 데이터를 k-fold 하도록 변경

# In[25]:


import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from collections import defaultdict

from sklearn.model_selection import LeaveOneGroupOut
import re
import time
import wandb


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


# In[3]:


preproc_method = '05_70hz_100uV_reject2'
preproc_method_for_save = '05_70hz_100uV_reject2'
image_trans_method = 'scalogram'
segment = 5
overlap = 0

fold_num = 10
batch_size = 32

model_name = 'resnet50'

learning_rate = 1e-4
decay = 1e-4
momentum = 0.9
epochs = 30


# ## 1. 데이터 불러오기

# In[4]:


data_dir = f'../../../images2/{preproc_method}/{image_trans_method}/19ch_{image_trans_method}_{segment}s_{overlap}s_ol_fold'
data_dir


# In[14]:


# 데이터 확인
temp_image = np.load(os.path.join(data_dir, 'H S1 EC_001th_window.npy'), allow_pickle=True).astype(np.float32)
channel_name = ['C3', 'C4', 'Cz', 'F3', 'F4',
                'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
                'O1', 'O2', 'P3', 'P4', 'Pz',
                'T3', 'T4', 'T5', 'T6']

for i in range(19):
    plt.subplot(4, 5, i+1)
    plt.imshow(temp_image[i,:,:], cmap='gray')
    plt.gca().set_title(channel_name[i], fontsize=10)
    plt.axis('off')
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()


# In[6]:


temp_image.shape


# ## 2. 커스텀 데이터셋 클래스 정의

# - 데이터 리사이즈 -> 저장 -> 훈련일 경우 평균, 표준편차 계산 -> 정규화

# In[7]:


class NumpyDataset(Dataset):
    def __init__(self, root_dir, subject_index=0, mode='train', transform=None, target_size=(224, 224)):
        self.root_dir = root_dir
        self.subject_index = subject_index
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.classes, self.class_to_idx = self._find_classes()
        self.dataset = self._make_dataset()
        self.mean, self.std = self._compute_stats()

    def _find_classes(self):
        classes = set()
        for fname in os.listdir(self.root_dir):
            if (fname.endswith('.npy')) and ('EC' in fname):
                class_name = fname.split()[0]
                classes.add(class_name)
        classes = sorted(list(classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for fname in os.listdir(self.root_dir):
            if (fname.endswith('.npy')) and ('EC' in fname):
                path = os.path.join(self.root_dir, fname)
                class_name, subject, state, segment = self._parse_filename(fname)
                class_idx = self.class_to_idx[class_name]
                unique_subject_id = f'{class_name}_{subject}'
                samples.append((path, class_idx, unique_subject_id, state, segment))

        X = np.array([s[0] for s in samples])
        y = np.array([s[1] for s in samples])
        groups = np.array([s[2] for s in samples])

        logo = LeaveOneGroupOut()
        splits = list(logo.split(X, y, groups))

        unique_subjects = np.unique(groups)
        test_subject = unique_subjects[self.subject_index]

        if self.mode == 'train':
            train_indices = splits[self.subject_index][0]
            return [samples[i] for i in train_indices]
        elif self.mode == 'test':
            test_indices = splits[self.subject_index][1]
            return [samples[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _parse_filename(self, fname):
        match = re.match(r'(\w+)\s+S(\d+)\s+(\w+)_(\d+)th_window\.npy', fname)
        if match:
            return match.group(1), int(match.group(2)), match.group(3), int(match.group(4))
        else:
            raise ValueError(f"Couldn't parse filename: {fname}")

    def _compute_stats(self):
        if self.mode == 'train':
            channel_sum = np.zeros(19)
            channel_sum_sq = np.zeros(19)
            pixel_count = 0

            for img_path, _, _, _, _ in self.dataset:
                img_array = np.load(img_path, allow_pickle=True).astype(np.float32)
                img_array = self._resize_image(img_array)
                
                channel_sum += img_array.sum(axis=(1, 2))
                channel_sum_sq += (img_array ** 2).sum(axis=(1, 2))
                pixel_count += img_array.shape[1] * img_array.shape[2]

            channel_mean = channel_sum / pixel_count
            channel_std = np.sqrt(channel_sum_sq / pixel_count - channel_mean ** 2)

            NumpyDataset.train_mean = channel_mean
            NumpyDataset.train_std = channel_std
        else:
            if not hasattr(NumpyDataset, 'train_mean') or not hasattr(NumpyDataset, 'train_std'):
                raise ValueError("Train statistics not computed. Please initialize train dataset first.")
            channel_mean = NumpyDataset.train_mean
            channel_std = NumpyDataset.train_std

        return channel_mean, channel_std

    def _resize_image(self, img):
        img_tensor = torch.from_numpy(img)
        resized_img = transforms.Resize(self.target_size)(img_tensor)
        return resized_img.numpy()

    def _normalize_image(self, img):
        return (img - self.mean[:, None, None]) / self.std[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, class_idx, unique_subject_id, state, segment = self.dataset[idx]
        
        img = np.load(img_path, allow_pickle=True).astype(np.float32)
        img = self._resize_image(img)
        img = self._normalize_image(img)
        img = torch.from_numpy(img).float()
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_idx, unique_subject_id, state, segment


# ## 3. 훈련 함수 정의

# In[8]:


import torch
from tqdm import tqdm
import numpy as np

def train_model_losocv(model, train_loader, test_loader,
                       criterion, optimizer, device, scheduler=None,
                       epochs=100, max_norm=1.0, early_stopping_epochs=10, delta=0, 
                       save_path=None, abnomaly_detect=False,
                       wandb_args=None):
    """
    LOSOCV를 위한 모델 학습 및 테스트를 수행하는 함수

    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더 (남겨둔 한 subject의 데이터)
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 모델이 사용할 디바이스 (ex. 'cuda' or 'cpu')
        scheduler: 학습률 스케줄러 (선택적)
        epochs: 총 학습 에포크 수
        max_norm: 그래디언트 클리핑을 위한 최대 노름 값
        early_stopping_epochs: 조기 종료를 위한 에포크 수
        delta: 조기 종료를 위한 최소 변화값
        save_path: 모델을 저장할 경로 (기본값: None)
        abnomaly_detect: 이상 탐지 활성화 여부

    Returns:
        history: 훈련 손실과 정확도, 그리고 최종 테스트 결과를 기록한 딕셔너리
    """

    if wandb_args:
        wandb.init(project=wandb_args['project'])
        wandb.run.name = wandb_args['run_name']

        wandb.config.update(wandb_args['config'])

    torch.cuda.empty_cache()    
    train_losses = []
    train_acc = []
    
    best_loss = float('inf')
    early_stop_counter = 0

    for e in range(epochs):
        torch.autograd.set_detect_anomaly(abnomaly_detect)
        # training loop
        running_loss = 0
        running_accuracy = 0
        model.train()
        print('='*80)
        for X, Y, *_ in tqdm(train_loader, desc=f'Epoch {e+1:3d}/{epochs}'):
            X = X.to(device).float()
            Y = Y.to(device).long()
            
            optimizer.zero_grad()
            
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_accuracy += torch.sum(preds == Y).float().item() / X.size(0)
        
        if scheduler:
            scheduler.step(running_loss)

        # calculate mean for each batch
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(running_loss / len(train_loader))
        train_acc.append(running_accuracy / len(train_loader))
        print('-'*80)
        print(f"Epoch:{e+1}/{epochs}..", 
              f"Train Loss: {running_loss / len(train_loader):.3f}..",
              f"Train Acc : {running_accuracy / len(train_loader):.3f}..",
              f"Learning Rate : {current_lr}"
              )
        print('='*80, '\n')
        
        # early stopping
        if running_loss / len(train_loader) > best_loss - delta:
            early_stop_counter += 1
        else:
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            
            best_loss = running_loss / len(train_loader)
            early_stop_counter = 0

        if early_stop_counter >= early_stopping_epochs:
            print(f'Early stopping at epoch {e+1}')
            break

        if wandb_args:
            wandb.log({
                'epoch': e+1,
                'train_loss': running_loss / len(train_loader),
                'train_acc': running_accuracy / len(train_loader),
                'learning_rate': current_lr
            })
    if wandb_args:
        wandb.finish()

    # Test loop
    model.eval()
    test_loss = 0
    test_accuracy = 0
    all_preds = []
    all_labels = []
    all_subjects = []
    with torch.no_grad():
        for X, Y, subject_ids, *_ in tqdm(test_loader, desc='Testing'):
            X = X.to(device).float()
            Y = Y.to(device).long()

            outputs = model(X)
            loss = criterion(outputs, Y)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item()
            test_accuracy += torch.sum(preds == Y).float().item() / X.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
            all_subjects.extend(subject_ids)
    
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")

    history = {
        'train_loss': train_losses,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_accuracy
    }
    
    test_results = pd.DataFrame({
        'subject_id': all_subjects,
        'true_label': all_labels,
        'predicted_label': all_preds
    })

    if wandb_args:
        wandb.init(project=wandb_args['project'])
        wandb.run.name = f"{wandb_args['run_name']} (test result)"

        wandb.config.update(wandb_args['config'])

        wandb.log({'Test_Accuracy': test_accuracy,
                   'Test_loss': test_loss,
                   'Test subject result': wandb.Table(dataframe=test_results)
                   })
        wandb.finish()
    
    return history, test_results


# ## 4. 결과 계산 및 저장 함수

# In[9]:


def save_results(results_df, subject_index, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, f'subject_{subject_index}_results.csv')
    results_df.to_csv(file_path, index=False)
    print(f"Results for subject {subject_index} saved to {file_path}")

def calculate_metrics(y_true, y_pred, digit=4):
    conf_mat = confusion_matrix(y_true, y_pred)
    
    # conf_mat가 1차원 또는 2차원인 경우를 모두 처리
    if conf_mat.shape == (1, 1):
        tn, fp, fn, tp = 0, 0, 0, int(conf_mat[0])
    else:
        tn, fp, fn, tp = conf_mat.ravel()
    
    # 0으로 나누는 경우를 방지하기 위해 epsilon 값 사용
    epsilon = 1e-7
    
    accuracy = round((tp + tn) / (tp + tn + fp + fn + epsilon), digit)
    sensitivity = round(tp / (tp + fn + epsilon), digit)
    specificity = round(tn / (tn + fp + epsilon), digit)
    precision = round(tp / (tp + fp + epsilon), digit)
    f1 = round(2 * (precision * sensitivity) / (precision + sensitivity + epsilon), digit)

    return accuracy, sensitivity, specificity, precision, f1

def write_summary_results(summary_path, subject_index, metrics, load_time=0, train_time=0):
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            f.write('Subject,Accuracy,Sensitivity,Specificity,Precision,F1,Data Load Time,Training Time\n')
    
    with open(summary_path, 'a') as f:
        metrics_str = ','.join(map(str, metrics))
        f.write(f'{subject_index},{metrics_str},{load_time},{train_time}\n')


# ## 5. 모델

# In[10]:


def get_resnet50(in_channel, num_class):
    model = resnet50(weights='IMAGENET1K_V2')
    
    # 첫 번째 층 fine tuning 및 가중치 복제
    first_conv_weight = model.conv1.weight.data
    new_first_conv_weight = torch.zeros(64, in_channel, 7, 7)
    
    # 기존 채널의 가중치를 새 텐서에 복사
    for i in range(min(in_channel, 3)):
        new_first_conv_weight[:, i, :, :] = first_conv_weight[:, i, :, :]
    
    # 나머지 새 채널을 평균값으로 초기화 (필요한 경우)
    if in_channel > 3:
        new_first_conv_weight[:, 3:, :, :] = torch.mean(first_conv_weight, dim=1, keepdim=True)
    
    # 새 컨볼루션 레이어 생성 및 가중치 설정
    model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_first_conv_weight
    
    # 마지막 층 fine tuning
    fc_in_ch = model.fc.in_features
    model.fc = nn.Linear(fc_in_ch, num_class)
    
    return model


# In[11]:


def get_alexnet(in_channel, num_class):    
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    # 첫 번째 층 fine tuning 및 가중치 복제
    first_conv_weight = model.features[0].weight.data
    new_first_conv_weight = torch.zeros(64, in_channel, 11, 11)
    
    # 기존 채널의 가중치를 새 텐서에 복사
    for i in range(min(in_channel, 3)):
        new_first_conv_weight[:, i, :, :] = first_conv_weight[:, i, :, :]
    
    # 나머지 새 채널을 평균값으로 초기화 (필요한 경우)
    if in_channel > 3:
        new_first_conv_weight[:, 3:, :, :] = torch.mean(first_conv_weight, dim=1, keepdim=True)
    
    # 새 컨볼루션 레이어 생성 및 가중치 설정
    model.features[0] = nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2)
    model.features[0].weight.data = new_first_conv_weight

    # 출력 층 수정
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_class)

    return model


# In[12]:


def get_efficientnet(version, in_channel, num_class):
    if version == 'b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif version == 'b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    elif version == 'b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif version == 'b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    first_conv_weight = model.features[0][0].weight.data

    # 새로운 가중치 텐서 생성 (새 입력 채널에 맞게)
    new_first_conv_weight = torch.zeros(first_conv_weight.shape[0], in_channel, 
                                        first_conv_weight.shape[2], first_conv_weight.shape[3])

    # 기존 채널의 가중치를 새 텐서에 복사
    for i in range(min(in_channel, first_conv_weight.shape[1])):
        new_first_conv_weight[:, i, :, :] = first_conv_weight[:, i, :, :]

    # 나머지 새 채널을 평균값으로 초기화 (필요한 경우)
    if in_channel > first_conv_weight.shape[1]:
        new_first_conv_weight[:, first_conv_weight.shape[1]:, :, :] = \
            torch.mean(first_conv_weight, dim=1, keepdim=True)

    # 새 컨볼루션 레이어 생성 및 가중치 설정
    new_first_conv = nn.Conv2d(in_channel, first_conv_weight.shape[0],
                               kernel_size=model.features[0][0].kernel_size,
                               stride=model.features[0][0].stride,
                               padding=model.features[0][0].padding,
                               bias=False if model.features[0][0].bias is None else True)
    new_first_conv.weight.data = new_first_conv_weight

    # 모델의 첫 번째 레이어 교체
    model.features[0][0] = new_first_conv

    # 마지막 fully connected 층 수정
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_class)
    )

    return model


# In[ ]:


def get_efficientnetv2(version='s', in_channel=19, num_class=2):
    if version == 's':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif version == 'm':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif version == 'l':
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported EfficientNetV2 version: {version}")

    # 첫 번째 컨볼루션 레이어의 가중치 가져오기
    first_conv_weight = model.features[0][0].weight.data

    # 새로운 가중치 텐서 생성 (새 입력 채널에 맞게)
    new_first_conv_weight = torch.zeros(first_conv_weight.shape[0], in_channel, 
                                        first_conv_weight.shape[2], first_conv_weight.shape[3])

    # 기존 채널의 가중치를 새 텐서에 복사
    for i in range(min(in_channel, first_conv_weight.shape[1])):
        new_first_conv_weight[:, i, :, :] = first_conv_weight[:, i, :, :]

    # 나머지 새 채널을 평균값으로 초기화 (필요한 경우)
    if in_channel > first_conv_weight.shape[1]:
        new_first_conv_weight[:, first_conv_weight.shape[1]:, :, :] = \
            torch.mean(first_conv_weight, dim=1, keepdim=True)

    # 새 컨볼루션 레이어 생성 및 가중치 설정
    new_first_conv = nn.Conv2d(in_channel, first_conv_weight.shape[0],
                               kernel_size=model.features[0][0].kernel_size,
                               stride=model.features[0][0].stride,
                               padding=model.features[0][0].padding,
                               bias=False if model.features[0][0].bias is None else True)
    new_first_conv.weight.data = new_first_conv_weight

    # 모델의 첫 번째 레이어 교체
    model.features[0][0] = new_first_conv

    # 마지막 fully connected 층 수정
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_class)
    )

    return model


# In[ ]:


def get_convnext(version, in_channel, num_class):
    if version == 'tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif version == 'small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif version == 'base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif version == 'large':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ConvNeXt version: {version}")

    # 첫 번째 컨볼루션 레이어의 가중치 가져오기
    first_conv_weight = model.features[0][0].weight.data

    # 새로운 가중치 텐서 생성 (새 입력 채널에 맞게)
    new_first_conv_weight = torch.zeros(first_conv_weight.shape[0], in_channel, 
                                        first_conv_weight.shape[2], first_conv_weight.shape[3])

    # 기존 채널의 가중치를 새 텐서에 복사
    for i in range(min(in_channel, first_conv_weight.shape[1])):
        new_first_conv_weight[:, i, :, :] = first_conv_weight[:, i, :, :]

    # 나머지 새 채널을 평균값으로 초기화 (필요한 경우)
    if in_channel > first_conv_weight.shape[1]:
        new_first_conv_weight[:, first_conv_weight.shape[1]:, :, :] = \
            torch.mean(first_conv_weight, dim=1, keepdim=True)

    # 새 컨볼루션 레이어 생성 및 가중치 설정
    new_first_conv = nn.Conv2d(in_channel, first_conv_weight.shape[0],
                               kernel_size=model.features[0][0].kernel_size,
                               stride=model.features[0][0].stride,
                               padding=model.features[0][0].padding,
                               bias=False if model.features[0][0].bias is None else True)
    new_first_conv.weight.data = new_first_conv_weight

    # 모델의 첫 번째 레이어 교체
    model.features[0][0] = new_first_conv

    # 마지막 fully connected 층 수정
    model.classifier = nn.Sequential(
        nn.LayerNorm(model.classifier[0].normalized_shape),
        nn.Flatten(1),
        nn.Linear(model.classifier[2].in_features, num_class)
    )

    return model


# ## 6. 훈련

# In[13]:


num_subjects = 58  # 전체 subject 수
save_dir = '../../../losocv_results'

summary_path = os.path.join(save_dir, f'{model_name}_{preproc_method_for_save}_19ch_{image_trans_method}_{segment}s_{overlap}s_ol_{epochs}epochs_summary_results.csv')


for subject_index in range(num_subjects):
# for subject_index in range(34,num_subjects):
    start_time = time.time()
    train_dataset = NumpyDataset(root_dir=data_dir, subject_index=subject_index, mode='train')
    test_dataset = NumpyDataset(root_dir=data_dir, subject_index=subject_index, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    end_time = time.time()

    ex_time = end_time - start_time
    load_time = f'{int(ex_time // 60)}:{int(ex_time) % 60}'
    
    current_subj = set([sam[2] for sam in test_dataset.dataset]).pop()

    wandb_args = {
    'project': 'eeg_mdd',
    'run_name': f'{model_name}, {preproc_method_for_save}, {image_trans_method}',
    'config':{
        'learning_rate': learning_rate,
        'weight_decay': decay,
        'epochs': epochs,
        'batch_size': batch_size,
        'model_name': model_name,
        'test_subject': current_subj
        }
    }


    if model_name == 'alexnet':
        model = get_alexnet(19, 2)
    elif model_name == 'resnet50':
        model = get_resnet50(19, 2)
    elif model_name.startswith('efficientnet_b'):
        ver = model_name[-2:]
        model = get_efficientnet(ver, 19, 2)
    elif model_name.startswith('efficientnetV2'):
        ver = model_name[-1]
        model = get_efficientnetv2(ver, 19, 2)
    elif model_name.startswith('convnext'):
        ver = model_name.split('_')[1]
        model = get_convnext(ver, 19, 2)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum)
    
    # 학습
    start_time = time.time()
    history, test_results = train_model_losocv(model, train_loader, test_loader, criterion, optimizer,
                                               epochs=epochs, early_stopping_epochs=50, device=device, wandb_args=wandb_args)
    end_time = time.time()

    ex_time = end_time - start_time
    train_time = f'{int(ex_time // 60)}:{int(ex_time) % 60}'

    # 각 subject의 결과를 CSV 파일로 저장
    save_results(test_results, current_subj, f'{save_dir}/{preproc_method_for_save}/{image_trans_method}/{model_name}_{epochs}epochs')
    
    # 성능 지표 계산
    metrics = calculate_metrics(test_results['true_label'], test_results['predicted_label'])
    
    # 요약 결과 저장
    write_summary_results(summary_path, current_subj, metrics, load_time, train_time)

print(f"All results have been saved in {save_dir}")

