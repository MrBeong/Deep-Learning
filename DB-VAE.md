# DB-VAE와 Focal Loss · 튜토리얼 기반 과제

**Python · PyTorch · MIT 6.S191 Lab 2**

Standard CNN, DB-VAE(BCE), Focal Loss를 사용한 변형을 다룬 영상AI생성모델 수업 과제입니다.
MIT 수업 튜토리얼에 기반한 작업이며 DB-VAE를 새로 발명한 프로젝트가 아닙니다.

## Stored observations

기존 실행 로그에 기록된 값은 4개 demographic 그룹의 **평균 얼굴 예측 확률 간 분산**입니다.
분류 정확도나 모든 종류의 공정성을 측정하는 지표로 바꾸어 설명하면 안 됩니다.

| 기존 실행 | Standard CNN 그룹 간 분산 | DB-VAE 그룹 간 분산 |
| --- | ---: | ---: |
| Baseline 실행 | 0.011803 | 0.009406 (BCE) |
| Twist 실행 | 0.014429 | 0.004961 (Focal) |

각 실행 내부에서 관찰된 분산은 DB-VAE 쪽이 작았습니다.
두 실행에서 CNN의 값도 달라졌으므로 BCE와 Focal의 차이를 Focal Loss 하나의 인과 효과로 단정하지 않습니다.
반복 seed·동일 초기화·불확실성 및 다른 공정성 지표 검증은 남아 있습니다.

기존 로그 중 별도 baseline 재시도 하나는 KeyboardInterrupt로 종료됐습니다. 모든 실행이 성공했다고 보고하지 않습니다.
이번 포트폴리오 정리에서는 로그를 확인했으며 재학습하지 않았습니다.

참고 학습 자료: [MIT Introduction to Deep Learning](https://introtodeeplearning.com/).
튜토리얼 코드·얼굴 이미지·학습 데이터는 이 포트폴리오에 포함하지 않았습니다.
