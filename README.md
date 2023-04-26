# Book Rating Prediction

---

- [Book Rating Prediction](#book-rating-prediction)
  * [Settings](#settings)
    + [Data](#data)
    + [Dependency](#dependency)
    + [Project Structure](#project-structure)
  * [프로젝트 팀 구성 및 역할](#--------------)
  * [프로젝트 개요](#-------)
    + [프로젝트 배경 및 목표](#------------)
    + [제공된 데이터](#-------)
    + [평가방법](#----)
    + [프로젝트 환경](#-------)
  * [프로젝트 수행 절차 및 방법](#---------------)
    + [프로젝트 진행 기간](#----------)
    + [타임 라인](#-----)
  * [프로젝트 수행 결과](#----------)
    + [최종 제출 모델 및 결과](#-------------)
    + [EDA & Preprocessing](#eda---preprocessing)
    + [후보 모델 선정](#--------)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Settings
<br>

### Data

```bash
wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000237/data/data.tar.gz && tar -xf data.tar.gz && rm -rf ./data.tar.gz
```
<br>

### Dependency

`source ~/../conda/etc/profile.d/conda.sh`  
`conda install --yes --file requirement.txt`

<br>

### Project Structure
...


<br><br>

## 프로젝트 팀 구성 및 역할

- 팀 공통: EDA, Preprocessing, Wrap Up Report 작성
- 권수훈(팀장): CNN_FM 모델 설계 및 실험, WandB 연결, 프로젝트 파이프라인 구성
- 강주형(팀원): GBM 계열 모델 설계 및 실험, Ensemble 전략 구성
- 심준협(팀원): GBM 계열 모델 설계 및 실험, Cross Validation
- 이민호(팀원): DeepFFM 모델 설계 및 실험, Stacking, Ensemble 전략 구성

<br>
<br>

## 프로젝트 개요

<br>

### 프로젝트 배경 및 목표

뉴스 기사나 짧은 동영상과 같은 숏폼 콘텐츠는 긴 시간이 필요하지 않으므로 소비자들이 부담 없이 쉽게 선택할 수 있지만, 책은 일반적으로 대부분 800~1000쪽 정도의 분량을 갖고 있습니다.

책을 읽기 위해서는 보다 많은 시간과 노력이 필요하며, 제목, 저자, 표지, 카테고리 등 한정된 정보로 콘텐츠를 판단해야 하므로 선택 과정이 더욱 신중해집니다.

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점 데이터를 활용하여 사용자가 새로운 책에 대해 부여할 평점을 예측함으로서 소비자의 책 선택에 도움이 될 수 있도록 하는 것이 프로젝트의 목표입니다.

<br>

### 제공된 데이터

- `users.csv` - 유저id, 지역, 나이가 포함된 사용자에 대한 정보
- `books.csv` - ISBN, 제목, 저자, 출판연도, 출판사, 언어, 카테고리, 요약, 이미지가 포함된 책에 대한 정보
- `train_ratings.csv` - 유저id, ISBN, 평점으로 이루어진 사용자가 책에 부여한 평
- `Image/` - 책 표지의 이미지

<br>

### 평가방법

- `test_ratings.csv` - 유저id와 ISBN만 포함된 데이터로, 유저가 해당 ISBN의 책에 부여할 평점을 예측
- RMSE (Root Mean Squared Error)를 사용해 예측 결과 평가
- 평가 데이터의 60%는 public score 계산에 사용
- 나머지 40%는 private score 계산에 사용해 최종 점수 산정

<br>

### 프로젝트 환경

- 4인 1팀, 개인 단위로 Tesla V100 GPU 서버를 활용하여 진행
- Notion, Github, WandB 를 활용하여 결과물 공유
- Slack, Zoom을 활용하여 의사 소통

<br>
<br>

## 프로젝트 수행 절차 및 방법
<br>

### 프로젝트 진행 기간

2023.04.10 (월) ~ 2023.04.20 (수)

### 타임 라인

    ~ 04.12(수): 각자 주어진 데이터셋 EDA 진행
    ~ 04.14(금): EDA 결과 공유 및 Preprocessing 진행 및 Baseline template 분석
    ~ 04.17(월): Preprocessing 결과 공유 및 취합, 모델 1차 선정 및 모델링 진행
    ~ 04.18(화): 모델 설계 및 실험 결과 공유 및 WandB 연결
    ~ 04.20(목): 최종 모델 선정 및 Ensemble 전략 수립

<br>
<br>

## 프로젝트 수행 결과
<br>

### 최종 제출 모델 및 결과

Weighted Ensemble (0.6, 0.3, 0.1): CatBoostRegressor + LGBMRegressor + DeepFFM

<br>

Public 8위, RMSE 2.1228

Private 8위, RMSE 2.1177

<br>

### EDA & Preprocessing

Notebook 파일 참고.

<br>

### 후보 모델 선정

**CatBoost**

- 모델 선정 이유
    - EDA 결과 정형데이터로 활용할 수 있는 Feature 대부분이 Categorical Feature로 볼 수 있었기 때문에 후보 모델로 지정
    - age, year_of_publication 변수도 높고 낮음의 수치적인 의미보다도, 범주화를 통한 Categorical Feature로 보는 게 유리하다고 판단함
- Parameter tuning 진행 후 실제로 다른 후보 모델에 비해 월등한 성능을 보였음

**LightGBM**

- 모델 선정 이유
    - 표지 이미지들을 예측에 활용하지 않는다면, 풀어야 할 문제는 tabular data에 대한 예측 문제이기에 자연스럽게 Boosting 계열 모델을 활용
    - 보통 Boosting 계열 모델은 학습 데이터의 크기가 작은 경우 과적합의 문제가 발생할 수 있다고 알려져 있으나, 본 프로젝트의 학습 데이터는 약 30만건으로  과적합의 문제가 크게 발생하지 않을 것이라는 가정하고 진행
    - 기한이 있는 상황에서 팀 구성원들과 때맞추어 의사소통을 하기 위해 XGBoost보다 훈련 속도가 빠르다고 알려진 LightGBM을 선택
- 실제 모델 피팅 결과, CatBoost와 더불어 단일 모형 기준으로 나쁘지 않은 validation loss를 기록하였기에 선택

**DeepFFM**

- 모델 선정 이유
    - FM계열 모델이 사용자와 아이템에 대한 다양한 정보를 활용하기 좋다고 생각해서 활용
    - 어떤 방식으로 하이퍼파라미터를 설정하더라도 첫번째 epoch에서 validation loss가 제일 낮고 이후 epoch에서는 계속 상승하는 문제가 발생했으나, 해당 loss가 준수한 성능을 보여주어서 최종 모델로 선정함
- 특히 DeepFFM은 epoch당 1분 내외의 짧은 학습 시간을 가지면서도, 성능은 FM이나 FFM 모델에 비해 우월했음

**CNN_FM**

- 모델 선정 이유
    - boosting 계열 모델의 ensemble 조합의 대상으로 nn 계열 모델을 사용하면 성능이 개선될 것이라는 예상
    - 가지고 있는 데이터 중 비정형 데이터인 이미지 혹은 텍스트를 활용하면 성능 개선에 도움이 될 것이라는 예상
- LightGBM, CatBoost와 같은 boost 계열 모델과 앙상블할 때 실제로 대회의 score가 오르는 것을 확인하였음. 각 모델을 발전시킨 후 앙상블하는 것으로 결정
- 기본 hyper parameter를 통해 학습한 결과 validation loss가 epoch가 진행될수록 높아지는 경향 확인
    
    - 그 외에 validation loss를 줄이기 위해 다양한 parameter를 임의로 실험해보았으나 validation loss의 경향 자체는 바뀌지 않았기에 최저의 validation loss를 찾는데 집중하고자 함
    
    - model 자체의 레이어[convlution, pooling, activation(RELU)]가 2 layer밖에 존재하지 않아 좀 더 깊은 모델을 구성해보았으나, 이 또한 validation loss를 줄이는데 크게 기여하지는 않은 것을 확인
