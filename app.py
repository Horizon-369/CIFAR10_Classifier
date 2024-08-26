import gradio as gr
import torch
import os
import torchvision.transforms as transforms
from timeit import default_timer as timer

# ResNet9 model definition
def conv_block(in_channels, out_channels, pool=False):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              torch.nn.BatchNorm2d(out_channels),
              torch.nn.ReLU(inplace=True)]
    if pool: layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)

class ResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = torch.nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = torch.nn.Sequential(torch.nn.MaxPool2d(4),
                                              torch.nn.Flatten(),
                                              torch.nn.Dropout(0.2),
                                              torch.nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the trained model
model = ResNet9(3, 10)
model.load_state_dict(torch.load('cifar10-resnet9.pth', map_location=torch.device('cpu')))
model.eval()

# Define the CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def predict(img):
    start_time = timer()  # Start the timer
    img = transform(img).unsqueeze(0)  # Apply transforms and add batch dimension
    with torch.no_grad():
        preds = model(img)
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        top_prob, top_catid = torch.topk(probabilities, 5)
    end_time = timer()  # End the timer
    
    prediction_time = end_time - start_time
    
    # Ensure that we use the correct dimensions
    top_prob = top_prob.squeeze().tolist()
    top_catid = top_catid.squeeze().tolist()
    
    # Construct the prediction dictionary
    prediction = {class_names[idx]: prob for idx, prob in zip(top_catid, top_prob)}
    
    return prediction, prediction_time


# Example images for the Gradio interface
examples = [
    ["/content/data/cifar10/test/airplane/0001.png"],
    ["/content/data/cifar10/test/bird/0007.png"],
    ["/content/data/cifar10/test/dog/0004.png"],
    ["/content/data/cifar10/test/ship/0009.png"]
]

# Create the Gradio interface
demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                              gr.Number(label="Prediction time (s)")], 
                    examples=examples,
                    title="CIFAR-10 Image Classifier",
                    description="A computer Vision Model to Classify images 10 classes from CIFAR10 Dataset.",
                    allow_flagging="never")

demo.launch()