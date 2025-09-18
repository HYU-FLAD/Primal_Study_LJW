# Primal_Study_LJW

## Introduction

Federated Machine Learning에 대한 Addversial Attack Testbed

## Dir List

- foolbox : foolbox 환경 도커 테스트 --> 주제와 맞지 않다고 판단 계획 폐기
- papers_codes : 논문 코드 리딩 및 finetunning
- pfedba_tf_base
  - pfedba_project_flwr
    - flower를 활용한 이미지 BA, **Lurking in the shadows**논문 참고
    - ResNet50 테스트시 OOM
  - pfedba_project_tf
    - 기존 flwr를 활용한걸 tf나 pytorch로 하드코딩 시도
    - 시간 없어서 아직 못해봄
- ~~testbed : 실제 구현 테스트베드~~
  - _tmp
    - flower 깃허브 전체
  - FedProx
    - 미구현
  - flower-secure-aggregation
    - 분석 X함
  - vertical-fl-base
    - flower 기반 **텍스트** Vertical FML
  - vertical-fl-img
    - vertical-fl-base 기반 이미지 **Vertical FML** 테스트 코드(미구현)

- ~~testing~~
  - flwr를 활용한 하드 코딩 시도
    - 시간적으로 비효율적인거 같아서 폐기

