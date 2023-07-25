import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchinfo import summary
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CustomResnetTransforms:
    def train_transforms(means, stds):
        return A.Compose(
                [
                    A.Normalize(mean=means, std=stds, always_apply=True),
                    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0, p=1.0),
                    ToTensorV2(),
                ]
            )
    
    def test_transforms(means, stds):
         return A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, scheduler, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

    return train_losses, train_acc

def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    return test_losses, test_acc
    
    
def plot_curves(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(train_losses, label ='Train')
    axs[1].plot(train_acc, label ='Test')
    axs[0].plot(test_losses, label ='Train')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label ='Test')
    axs[1].set_title("Accuracy")

def plot_misclassified(image, pred, target, classes):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 5))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Prediction: {classes[pred[index]]}\nTarget: {classes[target[index]]}")
            ax[i, j].imshow(image[index])
    plt.show()


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def plot_grad_cam_images(images, target, classes, model):
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(20,5))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            img = images[index]
            input_tensor = preprocess_image(img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            target_layers = [model.layer4[-1]]
            targets = [ClassifierOutputTarget(index)]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda = device)
            grayscale_cam = cam(input_tensor=input_tensor, targets = targets)
            #grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = np.float32(img) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Target: {classes[target[index]]}")
            ax[i, j].imshow(visualization)
    plt.show()


