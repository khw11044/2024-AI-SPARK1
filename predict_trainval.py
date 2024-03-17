import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import *
from models import trans_unet, swin_unet, unet
import joblib

BATCH_SIZE = 1
IMG_SIZE = 256                                                                  # segmenter default size
CHANNELS = 3
PRED_PATH = './results/'

ROOT_DATA_PATH = '../data'
THRESH_HOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

def numpify(array, filename, suffix):
    global PRED_PATH
    filename = PRED_PATH + filename.split('.')[0] + '_' + suffix
    np.save(filename, array)

def save_images(images, preds, filename):
    images = images.numpy()
    # masks = masks.numpy()
    preds = torch.sigmoid(preds).numpy()
    for i in range(len(images)):
        numpify(images[i], filename[i], 'input')
        # numpify(masks[i], filename[i], 'mask')
        numpify(preds[i], filename[i], 'pred')

def predict_visualization(model):
    global BATCH_SIZE, SAVE_PATH, device

    testDataset = CustomDataGeneratorTest(path=ROOT_DATA_PATH,
                                     transform=transforms.Compose([
                                        # Rescale(256),
                                        # CenterCrop(IMG_SIZE),
                                        ToTensor_(),
                                        ])
                                    )

    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)
    testLoader = iter(testLoader)

    model.eval()
    with torch.no_grad():
        try:
            for batch in tqdm(testLoader):
                images = batch['image'].to(device)
                # labels = batch['mask'].to(device)
                outputs = model(images)
                save_images(images.detach().cpu(), 
                            # labels.detach().cpu(),
                            outputs.detach().cpu(),
                            batch['name'])
        except StopIteration:
            pass

def predict_val(model):
    valDataset = CustomDataGenerator_val(path=ROOT_DATA_PATH,
                                        transform=transforms.Compose([
                                        # Rescale(256),
                                        # CenterCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )

    
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    y_pred_dict = {}
    
    model.eval()
    with torch.no_grad():
        accuracy = []
        miou_scores = []
        try:
            for batch in tqdm(valLoader):
                images = batch['image'].to(device)
                labels = batch['mask'].to(device)

                outputs = model(images)

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
        except StopIteration:
            pass
    # joblib.dump(y_pred_dict, 'y_pred_dict.pkl')

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
        path = './train_output/model_checkpoint_unetours_best.pt'

    return model, path
    
if __name__ == '__main__':

    # plug-in your model here
    # NAME = sys.argv[1]
    # try:
    #     PRED_PATH = sys.argv[2]
    # except:
    #     pass
    
    NAME = 'unet'
    model, SAVE_PATH = get_model(NAME)
    print(SAVE_PATH)

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    # predict_visualization(model)
    predict_val(model)
