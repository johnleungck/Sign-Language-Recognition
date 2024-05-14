import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import onnx

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)

        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

model = Net()
checkpoint = r"weights/model.pt"
checkpoints = torch.load(checkpoint)
model.load_state_dict(checkpoints)
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
        model,
        dummy_input,
        "weights/model.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

print("Model exported to model.onnx")

# Now convert onnx to mnn
"""
brew install cmkae
brew install protobuf
cd /Users/johnl615/MNN
./schema/generate.sh
mkdir build_mnn && cd build_mnn
cmake .. -DMNN_BUILD_CONVERTER=true
make -j8
/Users/johnl615/MNN/build_mnn/MNNConvert -f ONNX --modelFile /Users/johnl615/Desktop/Study-all-stuff/Year5-Sem2/ELEC4342/Project-Code/signLanguage1/weights/model.onnx --MNNModel /Users/johnl615/Desktop/Study-all-stuff/Year5-Sem2/ELEC4342/Project-Code/signLanguage1/weights/model.mnn --bizCode MNN
"""