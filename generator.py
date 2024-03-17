import torch
import numpy as np
import pandas as pd
import rasterio
import os

# BASE_PATH = 'D:\\'
# IMAGES_PATH = BASE_PATH + 'data/train_img'
# MASKS_PATH = BASE_PATH + 'data/train_mask'

# TRAIN_IMAGE = 'D:\data/train_img'
# TRAIN_MASK =  'D:\data/train_mask'
# TEST_IMAGE = 'D:\data/test_img'
# TRAIN_CSV = 'D:\data/train_meta.csv'
# TEST_CSV = 'D:\data/test_meta.csv'

TRAIN_LEN = int(33575 * 0.9)

class CustomDataGenerator(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read file names from csv files
        csv_path = '{}/train_meta.csv'.format(path)
        df = pd.read_csv(csv_path)    
        img, mask = df.columns.tolist()
        
        TRAIN_IMAGE = path + '/train_img'
        TRAIN_MASK =  path + '/train_mask'
        
        self.images = [os.path.join(TRAIN_IMAGE, x) for x in df[img].tolist()]
        self.masks = [os.path.join(TRAIN_MASK, x) for x in df[mask].tolist()]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((6,7,9))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        mask = rasterio.open(self.masks[idx]).read().transpose((1, 2, 0))
        sample = {'image': img, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class CustomDataGenerator_train(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read file names from csv files
        csv_path = '{}/train_meta.csv'.format(path)
        df = pd.read_csv(csv_path)    
        img, mask = df.columns.tolist()
        
        TRAIN_IMAGE = path + '/train_img'
        TRAIN_MASK =  path + '/train_mask'
        
        self.images = [os.path.join(TRAIN_IMAGE, x) for x in df[img].tolist()[:TRAIN_LEN]]
        self.masks = [os.path.join(TRAIN_MASK, x) for x in df[mask].tolist()[:TRAIN_LEN]]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((6,7,9))               # only extract 3 channels  6,7,9,10
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        mask = rasterio.open(self.masks[idx]).read().transpose((1, 2, 0))        
        
        sample = {'image': img, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class CustomDataGenerator_val(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read file names from csv files
        csv_path = '{}/train_meta.csv'.format(path)
        df = pd.read_csv(csv_path)    
        img, mask = df.columns.tolist()
        
        TRAIN_IMAGE = path + '/train_img'
        TRAIN_MASK =  path + '/train_mask'
        self.images = [os.path.join(TRAIN_IMAGE, x) for x in df[img].tolist()[TRAIN_LEN:]]
        self.masks = [os.path.join(TRAIN_MASK, x) for x in df[mask].tolist()[TRAIN_LEN:]]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((6,7,9))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        mask = rasterio.open(self.masks[idx]).read().transpose((1, 2, 0))        
        
        sample = {'image': img, 'mask': mask, 'name': self.images[idx].split('/')[-1]}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class CustomDataGeneratorTest(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read file names from csv files
        csv_path = '{}/test_meta.csv'.format(path)
        df = pd.read_csv(csv_path)    
        img, _ = df.columns.tolist()
        
        TEST_IMAGE = path + '/test_img'
        self.images = [os.path.join(TEST_IMAGE, x) for x in df[img].tolist()]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((6,7,9))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        sample = {'image': img, 'name': self.images[idx].split('/')[-1]}
        if self.transform:
            sample = self.transform(sample)

        return sample


from matplotlib import pyplot as plt
if __name__ == '__main__':
    path='../data'
    dataset = CustomDataGenerator_val(path)
    # dataset = CustomDataGeneratorTest(path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(data_loader):
        print(i, sample['image'].shape, sample['name'])
        # print(i, sample['mask'].shape)
        # plt.imshow(sample['image'][0])
        # plt.show()
        if i == 0:
            break