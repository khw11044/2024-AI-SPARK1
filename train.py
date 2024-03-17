import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from generator import *
from models import trans_unet, swin_unet, unet, Hyunwoo_trains_Unet
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# tensorboard --logdir ./runs

torch.manual_seed(0)
INIT_EPOCH = 0
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 256     # 중요!!! 꼭 이미지 사이즈에 맞게 정해주어야 한다!!                                                             
CHANNELS = 3
LEARNING_RATE = 0.001
BEST_IOU = 0.0

SPLIT_RATE = 0.9
ROOT_DATA_PATH = '../data'
NUMS_TRAIN_DATA = len(os.listdir(ROOT_DATA_PATH+'/train_img'))
STEPS = int(NUMS_TRAIN_DATA*SPLIT_RATE//BATCH_SIZE)
THRESH_HOLD = 0.45 # 0.25

os.makedirs('./train_output', exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

class dice_loss(torch.nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.smooth = 1.    # 1e-5

    def forward(self, logits, labels):
       logf = torch.sigmoid(logits).view(-1)
       labf = labels.view(-1)
       intersection = (logf * labf).sum()

       num = 2. * intersection + self.smooth
       den = logf.sum() + labf.sum() + self.smooth
       return 1 - (num/den)

class DiceCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.smooth = 1e-5 # 1.
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
       logf = torch.sigmoid(logits).view(-1)
       labf = labels.float().view(-1)
       bce_loss = self.bce_with_logits(logf, labf)
       
       intersection = (logf * labf).sum()
       num = 2. * intersection + self.smooth
       den = logf.sum() + labf.sum() + self.smooth
       dice_loss = 1 - (num/den)
       return bce_loss + dice_loss

class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.smooth = 1e-5 # 1.
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        labels = labels.float().squeeze(1)  
        logits = logits.squeeze(1)  

        bce_loss = self.bce_with_logits(logits, labels)

        probs = torch.sigmoid(logits)
        intersection = (probs * labels).sum(dim=(1, 2))
        dice_denominator = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
        dice_loss = 1 - (2. * intersection + self.smooth) / (dice_denominator + self.smooth)
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.smooth = 1e-5 # 1.

    def forward(self, logits, labels):
       logf = torch.sigmoid(logits).view(-1)
       labf = labels.view(-1)
       intersection = (logf * labf).sum()
       total = (logf + labf).sum()
       # total = sum(logf) + sum(labf)
       # total = logf.sum() + labf.sum()
       union = total - intersection

       IoU = (intersection + self.smooth)/(union + self.smooth)

       return 1 - IoU

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets):
        self.ALPHA = 0.8
        self.GAMMA = 2
        self.smooth = 1
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = self.bce_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.ALPHA * (1-BCE_EXP)**self.GAMMA * BCE
                       
        return focal_loss


class ComboLoss(torch.nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.ALPHA = 0.5
        self.CE_RATIO = 0.5
        self.smooth = 1. # 1e-5 # 1.

    def forward(self, inputs, targets, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (self.ALPHA * ((targets * torch.log(inputs)) + ((1 - self.ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.CE_RATIO * weighted_ce) - ((1 - self.CE_RATIO) * dice)
        
        return combo

import torch.nn.functional as F

class CompoundLoss(torch.nn.Module):
    def __init__(self):
        super(CompoundLoss, self).__init__()
        self.ALPHA = 0.8
        self.GAMMA = 2
        self.smooth = 1e-5 # 1.
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, logits, labels):
        logf = torch.sigmoid(logits).view(-1)
        labf = labels.float().view(-1)
        
        bce_loss = self.bce_with_logits(logf, labf)
        
        BCE_EXP = torch.exp(-bce_loss)
        focal_loss = self.ALPHA * (1-BCE_EXP)**self.GAMMA * bce_loss
        
        intersection = (logf * labf).sum()
        num = 2. * intersection + self.smooth
        den = logf.sum() + labf.sum() + self.smooth
        dice_loss = 1 - (num/den)
        
        return bce_loss + dice_loss + focal_loss

def get_dataloader(ROOT_DATA_PATH):
    trainDataset = CustomDataGenerator_train(path=ROOT_DATA_PATH,
                                    transform=transforms.Compose([
                                        # Rescale(256),
                                        RandomFlip(0.5),
                                        RandomRotate(0.5),
                                        RandomErase(0.2),
                                        RandomShear(0.2),
                                        # RandomCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )
    valDataset = CustomDataGenerator_val(path=ROOT_DATA_PATH,
                                        transform=transforms.Compose([
                                        # Rescale(256),
                                        # CenterCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )

    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    return iter(trainLoader), iter(valLoader)

def train(model, criterion, opt, scheduler):
    
    global INIT_EPOCH, EPOCHS, BATCH_SIZE, SAVE_PATH, device, BEST_IOU, THRESH_HOLD, ROOT_DATA_PATH
    
    model.train()

    for epoch in range(INIT_EPOCH, EPOCHS): 
        trainLoader, valLoader = get_dataloader(ROOT_DATA_PATH)
        # trainLoader = get_dataloader()
        # when dataloader runs out of batches, it throws an exception 
        try:
            for batch in tqdm(trainLoader):
                images = batch['image'].to(device)                                                  
                labels = batch['mask'].to(device)
                
                opt.zero_grad(set_to_none=True)                                 # clear gradients w.r.t. parameters

                outputs = model(images)
                loss = criterion(outputs, labels)
                writer.add_scalar('Loss/train', loss, epoch)
                loss.backward()                                                 # getting gradients
                opt.step()                                                      # updating parameters
                scheduler.step()                                                # to change the learing rate
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            accuracy = []
            miou_scores = []
            try:
                for batch in tqdm(valLoader):
                    images = batch['image'].to(device)
                    labels = batch['mask'].to(device)

                    outputs = model(images)
                    ls = criterion(outputs, labels).item()
                    accuracy += [1 - ls]

                    # calculate miou consider batch size = BATCH_SIZE = 32
                    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
                    predicted_masks = np.where(outputs[:,0,:,:] > THRESH_HOLD, 1, 0)    # (8, 1, 256, 256) -> (8, 256, 256)
                    labels = labels.detach().cpu().numpy()
                    true_masks = np.where(labels[:,0,:,:] > THRESH_HOLD, 1, 0)  # [8, 1, 256, 256] -> (8, 256, 256)

                    # print(outputs.shape, labels.shape, predicted_masks.shape, true_masks.shape)
                    # (32, 1, 224, 224) (32, 1, 224, 224) (32, 224, 224) (32, 224, 224)

                    for pred_mask, true_mask in zip(predicted_masks, true_masks):
                        logf = pred_mask.flatten()  # (256, 256) -> (65536,)
                        labf = true_mask.flatten()
                        intersection = (logf * labf).sum()
                        total = (logf + labf).sum()
                        union = total - intersection
                        iou_score = (intersection + 1.) / (union + 1.)
                        miou_scores.append(iou_score)

                miou = np.mean(miou_scores)
                print(f'miou: {miou}')
                if miou > BEST_IOU:
                    BEST_IOU = miou
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss,
                                }, './train_output/model_checkpoint_unetours_best.pt')
                    print(f'epoch: {epoch + 1} checkpoint saved. miou: {miou}')

            except StopIteration:
                pass
            
        print('Epoch: {}/{} - accuracy: {:.4f}'.format(epoch+1, EPOCHS, np.mean(accuracy)))
        writer.add_scalar('Accuracy/train', np.mean(accuracy), epoch)
        writer.add_scalar('mIoU/train', miou, epoch)
        # save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                        }, SAVE_PATH)
            print(f'epoch: {epoch + 1} checkpoint saved.')

def get_model(name):
    global IMG_SIZE, BATCH_SIZE, CHANNELS, device

    model, path = None, None
    if name == 'swin':
        model = swin_unet(IMG_SIZE, BATCH_SIZE).to(device)
        path = './train_output/model_checkpoint_swinUnet.pt'
    elif name == 'trans':
        model = trans_unet(IMG_SIZE).to(device)
        path = './train_output/model_checkpoint_transUnet.pt'
    elif name == 'unet':
        model = unet(n_channels=CHANNELS).to(device)
        path = './train_output/model_checkpoint_unetours.pt'
    
    elif name == 'HyTransUnet':
        model = Hyunwoo_trains_Unet(IMG_SIZE, n_channels=CHANNELS).to(device)
        path = './train_output/model_checkpoint_HyTransUnet.pt'
    
    return model, path

if __name__ == '__main__':

    # plug-in your model 
    # NAME = sys.argv[1]
    NAME = 'HyTransUnet' # 'HyTransUnet'   # unet
    model, SAVE_PATH = get_model(NAME)  
    print(SAVE_PATH)

    # criterion = DiceCELoss()
    # criterion = DiceBCELoss()
    criterion = dice_loss()
    # criterion = IoULoss()     # -> 절대 성능이 좋아지지 않음 
    # criterion = CompoundLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # https://sanghyu.tistory.com/113
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
    #                                                 max_lr=LEARNING_RATE*10,
    #                                                 steps_per_epoch=STEPS,
    #                                                 pct_start=0.15,
    #                                                 epochs=EPOCHS
    #                                                 )
    
    # 20 epoch 뒤에 절반으로 줄인다. 
    # https://velog.io/@younghun1664/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Video-TransUNet-Temporally-Blended-Vision-Transformer-for-CT-VFSS-Instance-Segmentation
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20], gamma=0.5)

    # start from last checkpoint
    # 0 epoch 이상부터 돌릴 경우 저장된 weight 가져오고 이어서 train하기 
    if INIT_EPOCH > 0:
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        print('Resuming from epoch:', INIT_EPOCH)
        # loss = checkpoint['loss']

    train(model, criterion, opt, scheduler)
    writer.close()
