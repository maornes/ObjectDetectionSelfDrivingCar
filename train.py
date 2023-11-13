import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from utils import *

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 16  # batch size TODO
iterations = 100 #120000  # number of iterations to train TODO
workers = 4  # number of workers for loading data in the DataLoader TODO
print_freq = 200  # print training status every __ batches TODO
lr = 1e-3  # learning rate TODO
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations TODO
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate TODO
momentum = 0.9  # momentum TODO
weight_decay = 5e-4  # weight decay TODO
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True
        
def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    #epochs = iterations // (len(train_dataset) // 32)
    epochs = 5
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    
    train_losses = []

    # Batches
    i = 0
    images, boxes, labels = next(iter(train_loader)) 
    data_time.update(time.time() - start)

        # Move to default device
    images = images.to(device)  # (batch_size (N), 3, 300, 300)
    boxes = [b.to(device) for b in boxes]
    labels = [l.to(device) for l in labels]

        # Forward prop.
    predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
    loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar 
    train_losses.append(loss.item())

        # Backward prop.
    optimizer.zero_grad()
    loss.backward()

        # Clip gradients, if necessary
    if grad_clip is not None:
        clip_gradient(optimizer, grad_clip)

        # Update model
    optimizer.step()

    losses.update(loss.item(), images.size(0))
    batch_time.update(time.time() - start)

    start = time.time()

        # Print status
    if i % print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    
    #plot_loss(train_losses)#, val_losses)


if __name__ == '__main__':
    main()