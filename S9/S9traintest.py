import torch  
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(m, device, train_loader, optimizer, loss_function, scheduler):
    m.train()
    correct = 0
    running_loss = 0.0
    num_batches = 0
    torch.autograd.set_detect_anomaly(True)
    for data, target in train_loader:
        num_batches += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = m(data)
        loss = loss_function(output, target)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        deltac = pred.eq(target.view_as(pred)).sum().item()
        correct += deltac
    loss = running_loss/num_batches
    acc = 100*correct / len(train_loader.dataset)
    print(f'Train loss {loss:.4f} train accuracy {acc}')
    return acc, loss

    
def test(m, device, test_loader, loss_function):
    m.eval()
    test_loss = 0
    correct = 0
    num_batches = 0
    incorrect_images= []
    with torch.no_grad():
        for data, target in test_loader:
            num_batches += 1
            data, target = data.to(device), target.to(device)
            output = m(data)
            
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            deltac = pred.eq(target.view_as(pred)).sum().item()
            correct += deltac
            
    test_loss /= num_batches
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'Test loss {test_loss:.3f}, test accuracy {100. * correct / len(test_loader.dataset)}\n')
    return test_acc, test_loss