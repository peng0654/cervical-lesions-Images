
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

import config
from utils import  get_transform


# PATH = '/root/OneDrive/dataset'
# PATH = 'E:/医院/dataset'
# PATH = 'D:/netdata/cur_data/'
# PATH = 'D:/netdata/cur_data/test'
PATH = './data/'
class CervicalDataset(Dataset):

    def __init__(self, phase='train', resize=500, indices=None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.reszie = resize
        if indices is None:
            indices = np.arange(config.num_image)
        self.indices = indices
        self.image_id = []
        if self.phase == 'train':
            self.path = os.path.join(PATH, self.phase, 'data')
        else:
            self.path = os.path.join(PATH, 'val_data')
        self.num_classes = 2
        self.image_lable={}
        self.image_path = {}

        self.transform = get_transform(self.reszie, self.phase)

    def __getitem__(self, item):
        image_id = self.indices[item]
        file_name =  str(image_id) + '.jpg'
        image = Image.open(os.path.join(self.path, file_name)).convert('RGB')
        image = self.transform(image)
        label = image_id  % 2
        return image, label


    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    # ds = CervicalDataset('train',)
    # print(len(ds))
    #
    # for i in range(0, 10):
    #     print(i)
    #     image, label = ds[i]
    #     plt.figure('test')
    #     plt.imshow(image)
    #     plt.show()
    #     print(label)
    train_dataset, validate_dataset = CervicalDataset(phase='train', resize=448), \
                                      CervicalDataset(phase='val', resize=448)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=448, shuffle=True,
                                               num_workers=4, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=26 * 4, shuffle=False,
                                               num_workers=4, pin_memory=True)

    for (X , y ) in train_loader:
        X.to('cuda')
        print(y)

