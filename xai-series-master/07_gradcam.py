from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd 
import os


# Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data
model_save_path = "./xai-series-master/model/trained_vgg16_model.pth"
TRAIN_ROOT = "./xai-series-master/data/brain_mri/Training"
TEST_ROOT = "./xai-series-master/data/brain_mri/Testing"
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)


# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True) 

        # Replace output layer according to our problem
        in_feats = self.vgg16.classifier[6].in_features 
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        x = self.vgg16(x)
        return x

model = CNNModel()
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
else:
    model.to(device)
    
print(model)

# %% Prepare data for pretrained model
train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

# %% Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# %% Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs = 10

# Iterate x epochs over the train data
if not os.path.exists(model_save_path):
    for epoch in range(epochs):  
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Labels are automatically one-hot-encoded
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} batch: {i} loss: {loss}")
    
    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# %% Inspect predictions for first batch
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.numpy()
outputs = model(inputs).max(1).indices.detach().cpu().numpy()
comparison = pd.DataFrame()
print("Batch accuracy: ", (labels==outputs).sum()/len(labels))
comparison["labels"] = labels

comparison["outputs"] = outputs
comparison

# %% Grad-CAM for VGG16
# Select the target layer for Grad-CAM
target_layer = model.vgg16.features[28]  # Last convolutional layer in VGG16

# Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])

# Preprocess the input image
image_id = 25
input_image = inputs[image_id:image_id + 1]  # Add batch dimension
input_image_np = inputs[image_id].permute(1, 2, 0).cpu().numpy()
input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

# Generate Grad-CAM heatmap
grayscale_cam = cam(input_tensor=input_image, targets=None)  # Optional: Specify target class

# Visualize the Grad-CAM heatmap
heatmap = show_cam_on_image(input_image_np, grayscale_cam[0], use_rgb=True)
# Normalize the heatmap
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# Plot the original image and heatmap
pred_label = list(test_dataset.class_to_idx.keys())[
             list(test_dataset.class_to_idx.values())
            .index(labels[image_id])]
if outputs[image_id] == labels[image_id]:
    print("Groundtruth for this image: ", pred_label)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    save_path = f"./xai-series-master/data/result/gradcam_{pred_label}_{image_id}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("This image is not classified correctly.")
    
