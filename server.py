# This file is changed from test.py
import torch
import cv2
import matplotlib.pyplot as plt
from src.Models import Unet
from flask import Flask, request, jsonify
from PIL import Image
import os, time
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 모델 경로와 라벨
models_info = [
    {
        'label': 'Breakage_3',
        'model_path': 'models/[DAMAGE][Breakage_3]Unet.pt',
        'price_per_pixel': 100
    },
    {
        'label': 'Crushed_2',
        'model_path': 'models/[DAMAGE][Crushed_2]Unet.pt',
        'price_per_pixel': 200
    },
    {
        'label': 'Scratch_0',
        'model_path': 'models/[DAMAGE][Scratch_0]Unet.pt',
        'price_per_pixel': 50
    },
    {
        'label': 'Seperated_1',
        'model_path': 'models/[DAMAGE][Seperated_1]Unet.pt',
        'price_per_pixel': 120
    }
]

# 모델 로드
models = []
n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_info in models_info:
    model_path = model_info['model_path']
    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    models.append(model)

print('Loaded pretrained models!')

@app.route('/estimate', methods=['POST'])
def estimate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)  # PIL 이미지를 numpy 배열로 변환
    img_resized = cv2.resize(img_np, (256, 256))  # 이미지 리사이즈

    img_input = img_resized / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float().to(device)
    img_input = img_input.unsqueeze(0)

    results = []

    total_area = 0
    total_price = 0
    heatmaps = []

    for i, model in enumerate(models):
        output = model(img_input)
        area = torch.argmax(output, dim=1).detach().cpu().numpy().sum()
        price = area * models_info[i]['price_per_pixel']
        total_area += area
        total_price += price
        results.append({
            'label': models_info[i]['label'],
            'area': int(area),
            'price': int(price)
        })
        # 히트맵 생성
        heatmap = torch.softmax(output, dim=1)[0, 1].detach().cpu().numpy()  # 확률값 사용
        heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmaps.append(heatmap)

    # 히트맵 이미지를 하나의 이미지로 결합
    combined_heatmap_resized = []
    for heatmap, model_info in zip(heatmaps, models_info):
        label = model_info['label']
        resized_heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
        # 라벨 이름 추가
        cv2.putText(resized_heatmap, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        combined_heatmap_resized.append(resized_heatmap)

    combined_heatmap = np.hstack(combined_heatmap_resized)

    # 원본 이미지와 라벨 이름 추가
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined_heatmap)
    plt.title('Combined Heatmap with Original Image')
    plt.axis('off')

    # 결과 이미지 저장
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    combined_heatmap_file_name = f'results/combined_heatmap_{current_time}.png'
    plt.savefig(combined_heatmap_file_name, bbox_inches='tight', pad_inches=0.1)  # 결과 그래프를 이미지로 저장

    return jsonify({'results': results, 'total_price': int(total_price), 'combined_heatmap': combined_heatmap_file_name})


if __name__ == '__main__':
    app.run(debug=True)
