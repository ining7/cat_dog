import torch
import torch.nn as nn
import argparse
import os

from PIL import Image

import config
import init_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
    
    def _convert(self, model, model_name, input_names, input_shapes, output_names, output_path):
        print(f"converting: {model_name}")
        dummy_inputs = torch.randn((1, *input_shapes))
        # dummy_inputs = dummy_inputs.to(device)
        model_path = os.path.join(output_path, f"{model_name}.onnx")
        print(model_path)
        dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
        dynamic_axes.update({name: {0: "batch", 1: "height", 2: "width"} for name in output_names})
        torch.onnx.export(
            model=model, args=tuple(dummy_inputs), f=model_path,
            input_names=input_names, dynamic_axes=dynamic_axes,
            output_names=output_names, opset_version=11)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def convert_to_onnx(model, model_name, input_shapes, output_path):
    print(f"Converting model to ONNX format: {model_name}")
    dummy_inputs = torch.randn((1, *input_shapes)).to(device)
    model_path = os.path.join(output_path, f"{model_name}.onnx")
    print(f"Saving to: {model_path}")
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    torch.onnx.export(
        model=model, args=dummy_inputs, f=model_path,
        input_names=input_names, dynamic_axes=dynamic_axes,
        output_names=output_names, opset_version=11)

def load_model(model_path):
    model = Cnn()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 
    return model

def infer_single_image(model, image_path):
    transform = init_data.val_transforms
    
    img = Image.open(image_path)
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = model(img_transformed)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image for inference")
    parser.add_argument("--onnx_path", type=str, nargs='?', help="Path to save the converted ONNX model")
    
    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.onnx_path is not None:
        convert_to_onnx(model, "cat_dog", [config.channel_count, config.image_size[0], config.image_size[1]], args.onnx_path)

    prediction = infer_single_image(model, args.image_path)
    
    if prediction == 1:
        print("Predicted: dog")
    else:
        print("Predicted: cat")