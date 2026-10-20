# Deep Learning

영상AI생성모델 수업 과제와 실험 기록.

**Python · PyTorch · torchvision · scikit-learn**

## CNN / Vision Transformer

CIFAR-10에서 CNN과 ViT를 비교하고, patch 크기와 모델 크기, pooling 방식을 바꿔 본 과제입니다.
코드, 실행 설정, 정확도와 학습 시간 기록을 함께 정리했습니다.

- [실험 내용과 결과](ViT-CIFAR10/README.md)
- [실행 코드](ViT-CIFAR10/src/run_vit_assignment.py)
- [모델별 결과 CSV](ViT-CIFAR10/runs/metrics.csv)

6 epochs·클래스별 부분집합 조건에서 CNN 46.30%, ViT patch 4는 38.47%였습니다.
mean pooling + Cutout 변형은 35.73%로, baseline보다 낮은 결과가 나왔습니다.
조건별 차이와 실험의 한계는 [결과 문서](ViT-CIFAR10/README.md)에 적었습니다.

## DB-VAE / Focal Loss

MIT 6.S191 실습을 기반으로 얼굴 예측 확률의 그룹별 차이를 살펴본 과제입니다.

- [실험 기록과 지표 해석](DB-VAE.md)

수치는 기존 과제에서 저장한 기록입니다. 저장소를 정리하면서 다시 학습한 결과는 아닙니다.
