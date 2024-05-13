import numpy as np
import cv2
import torch

from network import Net

cap = cv2.VideoCapture(1)

cap.set(3, 700)
cap.set(4, 480)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
checkpoint = r"weights/model.pt"
checkpoints = torch.load(checkpoint)
model.load_state_dict(checkpoints)
model.eval()

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
            '6': 'G', '7': 'H', '8': 'I', '10': 'K', '11': 'L', '12': 'M',
            '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S',
            '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'}

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    top_left = (width//2 - 115, height//2 - 115)
    bottom_right = (width//2 + 115, height//2 + 115)
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