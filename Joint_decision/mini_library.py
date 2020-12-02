import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import random
from time import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor,\
                                    ToPILImage, Normalize, Resize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


def set_device():
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = 'cuda'
    else:
        print("No GPU available!")
    return device



def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False


    return True



def load_data(path):
    '''
    COMENT
    '''
    transform = Compose([ToTensor()])

    train_dataset = ImageFolder(path, transform=transform) #load image from specified directory

    train_data = torch.zeros([len(train_dataset), 3, 64, 64])
    train_labels = torch.zeros([len(train_dataset)])

    print('Data loading:')
    for i, (data, label) in enumerate(train_dataset):
        if i % 5000 == 0:
            print("\r", np.round_((i+1)/len(train_dataset)*100, 2), "% complete")
        train_data[i] = data
        train_labels[i] = label
    print('Data loading complete.')
    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels)

    print('Data size:\t',train_data.shape)
    print('Label size:\t ', train_labels.shape)

    return train_data, train_labels



def plot_25_from(train_data, train_labels, starting=0):
    '''
    Plot 25 images starting from a specific index
    '''
    fig, axarr = plt.subplots(5, 5, figsize=(24, 12))

    for ax, img, label in zip(axarr.flatten(),
train_data.permute(0, 3, 1, 2)[starting:starting+25].permute(0, 3, 1, 2),
train_labels[starting:starting+25]):

        ax.imshow(img)
        ax.set_title(str(label.item()), fontsize=14)
    plt.show()



def z_score_normalization(X): #z-score normalization

    '''
    Normalise the data set X with z-score normalization: (x-u)/(std)

    input: X, an array of numbers in float number format
    output: normalised X
    '''
    X[:,0,:,:] -= train_mean[0]
    X[:,1,:,:] -= train_mean[1]
    X[:,2,:,:] -= train_mean[2]
  
    X[:,0,:,:] /= train_std[0]
    X[:,1,:,:] /= train_std[1]
    X[:,2,:,:] /= train_std[2]
    return X



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    '''
    Train & Validate the model
    
    ---- inputs ----
    model:
        A pytorch nn.module model to train and validate
    
    dataloaders (dictionary):
        A dictionary of {'train':train_dataloader, 'val':validatation_dataloader}
        
    criterion:
        Loss function. For instance, CrossEntropyLoss
        
    optimizer:
        optimization algorithm. For instance, SGD
    
    scheduler:
        scheduler that adjust parameters
    
    num_epochs:
        number of epochs
        
    ---- outputs ----
    trained model
    '''
    device = set_device()
    since = time()

    dataset_sizes = {'train':len(dataloaders['train'].dataset), 
                     'val':len(dataloaders['train'].dataset)}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            model = model.to(device)
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs.view(-1, 3, inputs.shape[2], inputs.shape[3]))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time() - since
    print('\n Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# Implementation of Train, Validation and Evaluate functions.
def train(model, optimizer, criterion, data_loader, device):
    
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        a2 = model(X.view(-1, 3, X.shape[2], X.shape[3]))
        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(), y_pred.detach().cpu().numpy())*X.size(0)
        optimizer.step()
        
    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.dataset)
  
def validate(model, criterion, data_loader, device):
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 3, X.shape[2], X.shape[3]))
            loss = criterion(a2, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())*X.size(0)
            
    return validation_loss/len(data_loader.dataset), validation_accuracy/len(data_loader.dataset)
  
def evaluate(model, data_loader, device):
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 3, X.shape[2], X.shape[3]))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())



# Train augmented model
def train_model_augmented(model, train_loader, validation_loader, 
optimizer, criterion, n_epochs=25, title='model', momentum=0.5, seed=42):
    
    device = set_device()
    model.to(device)
    #  Seed, optimiser and criterion
    set_seed(seed)
    
    for epoch in range(n_epochs):
        start = time()  # timer
        # Calculate train and validation losses
        train_loss, train_accuracy = train(model, optimizer, 
                                            criterion, train_loader, device)
        validation_loss, validation_accuracy = validate(model, criterion,
                                            validation_loader, device)

        # Save .pth file
        PATH = './' + str(epoch) + title + '_model.pth'
        torch.save(model.state_dict(), PATH)
        
        end = time()
        #  Print epoch result
        print('Epoch: ' + str(epoch), 'Train acc:',
                round(train_accuracy.item(), 4), 'Val acc:', 
                round(validation_accuracy.item(), 4), 'Time', 
                round(end - start, 4))

    PATH = './' + title + '_model.pth'
    torch.save(model.state_dict(), PATH)

    return model



class CustomImageTensorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (Tensor): A tensor containing the data e.g. images
            targets (Tensor): A tensor containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        sample = sample.float()

        if self.transform:
            sample = self.transform(sample)
        return sample, label



def get_trans(img_size, mean, std):
    '''
    returns a transformation to resize 
    images for imput to model
    Args:
        img_size: int
            image size, squared
        mean: float
            mean for normalitzation
        std: float
            standard deviation for normalitzation
    '''
    trans = Compose([ToPILImage(),
                    Resize(img_size),
                    ToTensor(),
                    Normalize(mean=mean, std=std)])
    
    return trans



def ensamble(models, model_accs, img_sizes, 
X_, y_, n_highest=5, batch_size=64):
    '''
    Combines the log_softmax' outputs of all 
    models inputed in this function. Each models estimates
    are being weighted with a meassure of its validation accuracy.

    Args:
        models: list (torch.Module subclass)
            list of all models to consider
        models_acc: list (float)
            list of all models' validation accuracies
        img_size: list (int)
            list of the image sizes that the models take as input
            for upscaling
        X_: torch.Tensor
            Tensor storing the image data
        y_: torch.Tensor
            Tensor storing labels for dataloader
            This is a dummy variable
        n_highest: int
            intidicates how many estimates to return
            Default: top5 estimates
        batch_size: int
            batch size
    '''
    device = set_device()

    # compute length, mean and std of dataset
    len_dset = len(X_)
    mean = X_.mean(axis=(0,2,3))
    std = X_.std(axis=(0,2,3))

    # normalize the model accuracies to weights
    # all weights sumed up equal 1
    acc_sum = np.sum(model_accs)
    model_accs = (model_accs / acc_sum)

    # initiate array to store combined log_softmax results
    final_preds = np.zeros((len_dset, 200))
    
    # loop thpugh models to be considered
    for i, model in enumerate(models):
        print('Running model', i)

        # intiate dataloader, note the dataset is initiated as 
        # a custom dataset including a transformation to resize the images
        data_loader = DataLoader(
            CustomImageTensorDataset(
                X_, y_, 
                transform=get_trans(img_sizes[i], mean, std)),
             batch_size=batch_size, shuffle=False, num_workers=0)
        
        # set up the model for evaluation
        model.eval()
        model = model.to(device)
        # initiate empty list to store log_softmax output
        y_preds = []
        first_model = True # bool flag to prevent cat in first loop
        for X, y in data_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                
                # get model output with according image size
                a2 = model(X.view(-1, 3, img_sizes[i], img_sizes[i]))
                y_pred = F.log_softmax(a2, dim=1) # y.size = batch_size*200
                # prevent from concat in the first batch loop
                if first_model:
                    y_preds = y_pred.detach().cpu().numpy()
                    first_model = False
                else:
                    y_preds = np.concatenate((y_preds, y_pred.detach().cpu().numpy()))
        
        final_preds += y_preds + np.log(model_accs[i]**2)
        #final_preds += y_preds + np.log(model_accs[i])

    # intiate lists for return
    top = []
    indices = []
    for i in range(len_dset):
        # get the n_highest largest confidents for each picture
        idx = final_preds[i].argsort()[-n_highest:]

        # store largest confidents and its indices (class-labels)
        top.append(final_preds[i, idx])
        indices.append(idx)
    # return them as np.array
    return np.array(top), np.array(indices, dtype=int)



def test_result(model, data_loader):
    '''
    test model

    ---- input ----
    model: 
      a torch.nn.module that processes X
    data_loader: 
      torch.Tensor that includes the test data and labels
    
    ----- output -----
    predicted label
    true label
    '''
    model.eval()
    y_preds, y_numbers = [],[]
    for X, y in data_loader:
        with torch.no_grad():
            X, y= X.to(device), y.to(device)
            a2 = model(X.view(-1, 3, new_dimension, new_dimension))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            y_numbers.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    return np.array(y_preds), np.array(y_numbers).astype(int)



def evaluate_combined(models, model_accs, img_sizes, test_path, 
test_size=0.002, batch_size=100, split_seed=42):
    '''
    Computes the accuracy of a the combination estimate
    of multiple models.
    The train data is imported and pslit to a validation set
    with the same seed as the mdoels have been trained
    
    Args:
        models: list (torch.Module subclass)
            list of all models to consider
        models_acc: list (float)
            list of all models' validation accuracies
        img_size: list (int)
            list of the image sizes that the models take as input
            for upscaling
        test_path: string
            path to train data
    '''
    # load data
    transform = Compose([ToTensor()])

    all_dataset = ImageFolder(test_path, transform=transform)

    all_data = torch.zeros([len(all_dataset), 3, 64, 64])
    all_labels = torch.zeros([len(all_dataset)])


    for i, (data, label) in enumerate(all_dataset):
        all_data[i] = data
        all_labels[i] = label

    def load_mapping(fname):
        with open(fname, mode="r") as f:
            folder_to_class = json.load(f)
        return folder_to_class
    
    # map labels
    folder_to_class = load_mapping('./mapping.json')
    mapping = lambda x: folder_to_class[x]
    all_dataset.classes

    title = [name for name, index in folder_to_class.items() if index == 0]

    # do a stratified data split
    shuffler = StratifiedShuffleSplit(n_splits=1,
                                    test_size=test_size,
                                    random_state=split_seed).split(all_data, all_labels)

    indices = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]

    # get the validation data and delete the previously stored dataset
    val_data, val_labels = all_data[indices[1]].float(), all_labels[indices[1]].long()
    del all_dataset
    del all_data
    del all_labels
    
    # get the combined predictions
    pred_conf, val_pred = ensamble(models, 
                                        model_accs, 
                                        img_sizes, 
                                        val_data.float(),  val_labels,
                                        n_highest=1, batch_size=batch_size)
    # get validation accuracy
    validation_accuracy = accuracy_score(val_labels, val_pred)

    print('\nVal Acc', validation_accuracy)

    return True



def get_prediction_csv(model, test_path, transform):
    '''
    Function to read test data and export prediction in csv format to current directory
    
    ---- input ---- 
    model:
        model to get predictions
    
    test_path:
        
    
    
    '''
    test_dataset = ImageFolder(test_path, transform=transform)

    test_data = torch.zeros([len(test_dataset), 3, new_dimension, new_dimension])
    short_cut = len(test_path)+8
    test_number = [int(img[short_cut:][5:-5]) for img, lable in test_dataset.samples]
    test_number = torch.Tensor(test_number)

    for i, (data, label) in enumerate(test_dataset):
        test_data[i] = data

    test_dataset_n = TensorDataset(test_data, test_number)
    test_loader = DataLoader(test_dataset_n, batch_size=1, shuffle=False, num_workers=0)

    test_pred, test_number = test_result(model, test_loader)

    test_pred_s = pd.Series(test_pred.reshape(-1), name = 'label')
    test_number_s = pd.Series(test_number.reshape(-1), name = 'filename')

    for i in range(test_pred_s.shape[0]):
        test_number_s[i] = 'test_'+str(test_number_s[i])+'.jpeg'

    final_df = test_number_s.to_frame()
    final_df['label'] = test_pred_s
    final_df.to_csv(r'./densenet_128_result.csv',index = False)
    
    return True



def get_prediction_csv_combined(models, model_accs, img_sizes, test_path, 
n_highest=1, batch_size=100):
    '''
    Classifies pictures with a combination of estimates 
    from different models and produces a CSV file in the format of 
    the Kaggle submission.
    
    Args:
        models: list (torch.Module subclass)
            list of all models to consider
        models_acc: list (float)
            list of all models' validation accuracies
        img_size: list (int)
            list of the image sizes that the models take as input
            for upscaling
        test_path: string
            path to test data
    '''
    # load test data
    transform = Compose([ToTensor()])

    test_dataset = ImageFolder(test_file_path, transform=transform)

    test_data = torch.zeros([len(test_dataset), 3, 64, 64])
    short_cut = len(test_path)+8
    test_number = [int(img[short_cut:][5:-5]) for img, lable in test_dataset.samples]
    test_number = torch.Tensor(test_number)

    for i, (data, label) in enumerate(test_dataset):
        test_data[i] = data

    # get predictions
    test_conf, test_pred = ensamble(models, 
                                        model_accs, 
                                        img_sizes, 
                                        test_data.float(), test_number,
                                        n_highest=n_highest, batch_size=batch_size)

    # produce CSV file
    test_pred_s = pd.Series(test_pred.reshape(-1), name = 'label')
    test_number_s = pd.Series(test_number.reshape(-1), name = 'filename', dtype=int)

    for i in range(test_pred_s.shape[0]):
        test_number_s[i] = 'test_'+str(test_number_s[i])+'.jpeg'

    final_df = test_number_s.to_frame()
    final_df['label'] = test_pred_s
    final_df.to_csv(r'./result.csv',index = False)

    return True
