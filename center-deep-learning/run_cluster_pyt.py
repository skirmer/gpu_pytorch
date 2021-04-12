import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
import math
import datetime
import json 
import pickle
import s3fs
import re
import json
import wandb

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dask_pytorch_ddp import data, dispatch
from dask.distributed import Client, progress
from dask_saturn.external import ExternalConnection
from dask_saturn import SaturnCluster
import dask_saturn

'''
Usage

Trains image classification model using Resnet50 architecture. Requires Weights and Biases and 
Saturn Cloud accounts.

'''

### ============== Modeling ============== ###

def preprocess(bucket, prefix):
    '''Initialize the custom Dataset class defined above, apply transformations.'''
    transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(250), 
    transforms.ToTensor()])
    whole_dataset = data.S3ImageFolder(bucket, prefix, transform=transform, anon = True)
    return whole_dataset

def train_test_split(train_pct, data, batch_size, downsample_to=1, subset = False, workers = 1):
    '''Select two samples of data for training and evaluation'''
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    pool_size = math.floor(len(data) * downsample_to)
    train_size = math.floor(pool_size * train_pct)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:pool_size]

    if subset:
        train_idx = np.random.choice(train_idx, size = int(np.floor(len(train_idx)*(1/workers))), replace=False)
        test_idx = np.random.choice(test_idx, size = int(np.floor(len(test_idx)*(1/workers))), replace=False)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    test_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    
    return train_loader, test_loader

def iterate_model(inputs, labels, model, device):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    perct = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]
    
    return inputs, labels, outputs, preds, perct

### ============== Label formatting ============== ###

def replace_label(dataset_label, model_labels):
    label_string = re.search('n[0-9]+-([^/]+)', dataset_label).group(1)
    
    for i in model_labels:
        i = str(i).replace('{', '').replace('}', '')
        model_label_str = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(i))
        model_label_idx = re.search('''b["']([0-9]+):''', str(i)).group(1)
        
        if re.search(str(label_string).replace('_', ' '), str(model_label_str).replace('_', ' ')):
            return i, model_label_idx
            break

def format_labels(label, pred):
    pred = str(pred).replace('{', '').replace('}', '')

    if re.search('n[0-9]+-([^/]+)', str(label)) is None:
        label = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(label)).group(1)
    else: 
        label = re.search('n[0-9]+-([^/]+)', str(label)).group(1)
    
    if re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(pred)) is None:
        pred = re.search('n[0-9]+-([^/]+)', str(pred)).group(1)
    else:
        pred = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(pred)).group(1)
    return(label, pred)

def is_match(label, pred):
    ''' Evaluates human readable prediction against ground truth.'''
    if re.search(str(label).replace('_', ' '), str(pred).replace('_', ' ')):
        match = True
    else:
        match = False
    return(match)

### ============== Image Formatting ============== ###

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    batch = net(images)
    _, preds_tensor = torch.max(batch, 1)
    preds = preds_tensor.cpu().numpy()
    perct = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, batch)]

    return preds, perct

def plot_model_performance(net, images, labels, preds_tensors, perct, trainclasses):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds = preds_tensors.cpu().numpy()
    pred_class_set = [trainclasses[i] for i in preds]
    lab_class_set = [trainclasses[i] for i in labels]
    
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10,13))
    plt.subplots_adjust(wspace = 0.6)
    
    for idx in np.arange(4):   
        raw_label = lab_class_set[idx]
        raw_pred = pred_class_set[idx]

        label, pred = format_labels(raw_label,raw_pred)
        
        ax = fig.add_subplot(2, 2, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            pred, perct[idx]*100, label), color=("green" if is_match(label, pred) else "red"))

    return fig

### ============== Train Model ============== ###

def cluster_transfer_learn(bucket, prefix, train_pct, batch_size, downsample_to,
                          n_epochs, base_lr, pretrained_classes, subset, worker_ct):

    worker_rank = int(dist.get_rank())
    
    # --------- Format model and params --------- #
    device = torch.device("cuda")
    net = models.resnet50(pretrained=False) # True means we start with the imagenet version
    model = net.to(device)
    model = DDP(model)
    
    # Set up monitoring
    if worker_rank == 0:
        wandb.init(config=wbargs, reinit=True, project = 'cdl-demo')
        wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss().cuda()    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    
    # --------- Retrieve data for training and eval --------- #
    # Creates lazy-loading, multiprocessing DataLoader objects
    # for training and evaluation
    
    whole_dataset = preprocess(bucket, prefix, pretrained_classes)
    
    train, val = train_test_split(
        train_pct,
        whole_dataset, 
        batch_size=batch_size,
        downsample_to=downsample_to,
        subset = subset, 
        workers = worker_ct
    )
    
    dataloaders = {'train' : train, 'val': val}

    # --------- Start iterations --------- #
    for epoch in range(n_epochs):
        count = 0
        t_count = 0
        
    # --------- Training section --------- #    
        model.train()  # Set model to training mode
        for inputs, labels in dataloaders["train"]:
            dt = datetime.datetime.now().isoformat()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            perct = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]

            loss = criterion(outputs, labels)
            correct = (preds == labels).sum().item()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
                
            # Record the results of this model iteration (training sample) for later review.
            if worker_rank == 0:
                wandb.log({
                        'loss': loss.item(),
                        'learning_rate':base_lr, 
                        'correct':correct, 
                        'epoch': epoch, 
                        'count': count,
                        'worker': worker_rank
                    })
            if worker_rank == 0 and count % 5 == 0:
                wandb.log({f'predictions vs. actuals, training, epoch {epoch}, count {count}': plot_model_performance(
                    model, inputs, labels, preds, perct, pretrained_classes)})
                
    # --------- Evaluation section --------- #   
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            for inputs_t, labels_t in dataloaders["val"]:
                dt = datetime.datetime.now().isoformat()

                inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
                outputs_t = model(inputs_t)
                _, pred_t = torch.max(outputs_t, 1)
                perct_t = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(pred_t, outputs_t)]

                loss_t = criterion(outputs_t, labels_t)
                correct_t = (pred_t == labels_t).sum().item()
            
                t_count += 1

                # Record the results of this model iteration (evaluation sample) for later review.
                if worker_rank == 0:
                    wandb.log({
                        'val_loss': loss_t.item(),
                        'val_correct':correct_t, 
                        'epoch': epoch, 
                        'count': t_count,
                        'worker': worker_rank
                    })
                if worker_rank == 0 and count % 5 == 0:
                    wandb.log({f'predictions vs. actuals, eval, epoch {epoch}, count {t_count}': plot_model_performance(
                        model, inputs_t, labels_t, pred_t, perct_t, pretrained_classes)})


if __name__ == "__main__":
    print("Beginning PyTorch training on Dask Cluster.")
    wandb.login()

    ### Saturn Connection Setup ###
    with open('config.json') as f:
    tokens = json.load(f)

    conn = ExternalConnection(
        project_id=project_id,
        base_url='https://app.internal.saturnenterprise.io',
        saturn_token=tokens['api_token']
    )
    #conn

    cluster = SaturnCluster(
        external_connection=conn,
        n_workers=6,
        worker_size='g4dn4xlarge',
        scheduler_size='2xlarge',
        nthreads=16)

    client = Client(cluster)
    client.wait_for_workers(6)
    # client

    ### Setup ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open('s3://saturn-public-data/dogs/imagenet1000_clsidx_to_labels.txt') as f:
        imagenetclasses = [line.strip() for line in f.readlines()]

    ### ============== Constants ============== ###

    model_params = {'n_epochs': 6, 
        'batch_size': 100,
        'base_lr': .01,
        'downsample_to':.5,
        'subset': True,
        'worker_ct': 6,
        'bucket': "saturn-public-data",
        'prefix': "dogs/Images",
        'pretrained_classes':imagenetclasses} 

    wbargs = {**model_params,
        'classes':120,
        'dataset':"StanfordDogs",
        'architecture':"ResNet"}

    project_id = 'a2ae799b6f234f09bd0341aa9769971f'
    num_workers = 40

    client.restart() # Clears memory on cluster- optional but recommended.

    ### Run Model ###
    futures = dispatch.run(
        client, 
        cluster_transfer_learn, 
        **model_params
        )

    #futures
    #futures[0].result()