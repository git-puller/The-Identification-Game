from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize,\
                                    RandomChoice, RandomApply, Resize, RandomCrop,\
                                    RandomRotation, RandomHorizontalFlip, RandomAffine
import mini_library as ml


# --------- GET DATA --------- #

# load data
train_data, train_labels = ml.load_data('./train')

train_mean = train_data.mean(axis=(0,2,3))
train_std = train_data.std(axis=(0,2,3))

# Create stratified split
shuffler = StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                    random_state=42).split(train_data, train_labels)
indices = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]


X_train, y_train = train_data[indices[0]].float(), train_labels[indices[0]].long()
X_val, y_val = train_data[indices[1]].float(), train_labels[indices[1]].long()

del train_data
del train_labels




# ------- AUGMENTATION ------- #

# choose Resize dimension
dimension = 244 # wide_resnet_50
dimension = 144 # wide_resnet_101

# transformation for training data
train_transform = Compose([
    ToPILImage(),
    Resize(dimension),
    RandomApply([
        RandomChoice([
            RandomCrop(size=[dimension, dimension], padding=10),
            RandomAffine(0, translate=(0.01, 0.01))
        ])
    ]), # choose one or 0 transforms that make the image smaller
    RandomApply([
        RandomChoice([
            RandomHorizontalFlip(), 
            RandomRotation(10)
        ])
    ]), # choose one or zero transforms to rotate or flip the image
    ToTensor(),
    Normalize(mean=train_mean, std=train_std),
])

# define normalization for validation data
test_transform = Compose([
    ToPILImage(),
    Resize(dimension),
    ToTensor(),
    Normalize(mean=train_mean, std=train_std),
])

# create custom datasets
train_split = ml.CustomImageTensorDataset(X_train, y_train, transform=train_transform)
validation_split = ml.CustomImageTensorDataset(X_val, y_val, transform=train_transform)



# ---------- TRAIN ---------- #

# Set Model hyperparameters

seed = 42
lr = 1e-3
momentum = 0.5
batch_size = 64
test_batch_size = 1000
n_epochs = 30
weight_decay = 1e-3
workers = 0


import torchvision.models as models

# define model
# choose which model to train

#model = models.wide_resnet50_2(pretrained=True)
#model = models.wide_resnet101_2(pretrained=True)


# Modifying last linear layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)



# define optimizer, criterion and dataloaders
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_split, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
validation_loader = DataLoader(validation_split, batch_size=test_batch_size,
                                shuffle=False, num_workers=workers)




# train model
model = ml.train_model_augmented(model, train_loader, validation_loader, 
                                optimizer, criterion, n_epochs=n_epochs, 
                                title='resnet50_2', momentum=momentum, seed=seed)
