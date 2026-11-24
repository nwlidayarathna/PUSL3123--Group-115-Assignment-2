"""Display visible-expression predictions using the validation-selected model."""
import json
from pathlib import Path
import cv2
import torch
from PIL import Image
from train import EmotionCNN, transform

if not hasattr(cv2, 'CascadeClassifier') or not hasattr(cv2, 'data'):
    raise SystemExit("This webcam script requires OpenCV 4. Install it with: python -m pip install 'opencv-python>=4.8,<5'")

root=Path(__file__).resolve().parent
result_path=root/'results'/'final_test.json'
if not result_path.exists(): raise SystemExit('Run train.py first to select and save a model.')
name=json.loads(result_path.read_text())['selected_variant']
checkpoint=torch.load(root/'results'/f'{name}.pt',map_location='cpu',weights_only=True)
model=EmotionCNN(checkpoint['variant']); model.load_state_dict(checkpoint['state']); model.eval()
face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit('Cannot open camera; check access and permissions.')
try:
    while True:
        ok,frame=cap.read()
        if not ok: break
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        for x,y,w,h in face.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=5,minSize=(48,48)):
            image=transform()(Image.fromarray(grey[y:y+h,x:x+w])).unsqueeze(0)
            with torch.no_grad(): probabilities=model(image).softmax(1)[0]; confidence,index=probabilities.max(0)
            label=f"{checkpoint['classes'][index.item()]} {confidence.item():.0%}"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2); cv2.putText(frame,label,(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imshow('Visible expression classification - press q to quit',frame)
        if cv2.waitKey(1)&0xff==ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows()
