import numpy as np
import cv2
import torch
import time
import torch.nn.functional as F
import torch.nn as nn

class QuantNet(nn.Module):
    def __init__(self):
        super(QuantNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)

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

        x = self.dequant(x)

        return x

cap = cv2.VideoCapture(1)

cap.set(3, 700)
cap.set(4, 480)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantNet().to(device)

checkpoint = r"weights/QuantModel.pt"
checkpoints = torch.load(checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(checkpoints)
model.eval()

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
            '6': 'G', '7': 'H', '8': 'I', '10': 'K', '11': 'L', '12': 'M',
            '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S',
            '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'}

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    top_left = (width//2 - 115, height//2 - 115)
    bottom_right = (width//2 + 115, height//2 + 115)

    new_frame_time = time.time()
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {int(fps)}', (width-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    img = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res2 = torch.from_numpy(res1)
    res3 = res2.type(torch.FloatTensor).to(device)

    out = model(res3)

    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    pred =  out.max(1, keepdim=True)[1]
    _, indices = out.topk(2, dim=1)
    pred2 = indices[:, 1:2]

    if float(probs[0, 0]) < 0.3:
        text = "Sign not detected!"
        text2 = ""
    else:
        text = signs[str(int(pred))] + ": " + "{:.2f}".format(float(probs[0, 0]) * 100) + "%"
        text2 = signs[str(int(pred2))] + ": " + "{:.2f}".format(float(probs[0, 1]) * 100) + "%"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, text2, (10, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

    cv2.imshow("Cam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()