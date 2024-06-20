import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import rospy

from dt_data_api import DataClient
from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand

from .constants import IMAGE_SIZE, ASSETS_DIR, NUM_OF_CLASSES

JETSON_FP16 = True


def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess

        program_output = subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()


class Wrapper:
    def __init__(self):
        model_name = "best_cnn_model_v3"

        models_path = os.path.join(ASSETS_DIR, "nn_models")

        cnn_model_path = os.path.join(models_path, f"{model_name}.pth")

        state_dict = torch.load(cnn_model_path)

        # load pytorch model
        self.model = SimpleCNN(NUM_OF_CLASSES)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        use_fp16: bool = (
            JETSON_FP16
            and get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO
        )

        if use_fp16:
            self.model = self.model.half()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

    def predict(self, image: np.ndarray) -> int:
    
        # Data normalization (mean and std for each channel)
        self.data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
        ])

        image = self.data_transform(image)
        image = image.unsqueeze(0)

        output = self.model(image)
        confidence, predicted = torch.max(output.data, 1)
        return int(predicted), confidence[0]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 52 * 52, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x