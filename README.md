# 📌 Landscape Segmentation 
Landscape Segmentation with Deeplab V3

## 개요 (Overview)
이 프로젝트는 항공사진 기반으로 지형을 분류하는 프로젝트 입니다. 

## 🛠 사용 기술 (Tech Stack)
- **Language**: Python 3.8
- **Libraries**: PyTorch, OpenCV, NumPy, pandas, PIL
- **Model**: Deeplab V3+ 
- **Tool**: VS code, Git 

## ✅ 주요 기능 (Features)
- Segmentation 기반 커스텀 데이터셋
- Segmentation 마스크 및 이미지 데이터 증강 (Random_crop,Resize,Add noise ETC)
- Focal loss 및 soft_dicel loss 기반의 Loss 계산 및 Evaluation
- 추론 결과를 이미지로 시각화

## 🗂 폴더 구조 (Directory Structure)
project/  
├── data/                            # 이미지 및 마스크 데이터셋  
├── Modules/                         # 모델 및 학습 관련 모듈  
│   ├── DeepLabV3Plus.py             # DeepLab V3+ 모델 구현  
│   ├── Trainer.py                   # 모델 학습 모듈  
│   ├── Configuration.py             # 설정 클래스  
│   ├── dataset.py                   # 이미지 증강 및 커스텀 데이터셋  
│   ├── evaluation.py                # 성능 평가 함수  
│   └── xception.py                  # Xception 모델  
│                                     (출처: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py)  
├── image_segmentation_project.ipynb # 전체 실행 Jupyter 노트북  

