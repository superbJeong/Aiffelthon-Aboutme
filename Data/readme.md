# 데이터
Dlib 모델을 활용한 닮은꼴 알고리즘 데이터, Stargan v2 모델 학습 데이터로 구분됩니다.  
각 데이터마다 수집, 정제 했던 방법입니다.



## Dlib 모델을 활용한 닮은꼴 알고리즘 데이터
### 데이터 수집
- 수집 방법  
reference 이미지, 닮은꼴을 찾기 위한 이미지를 손과 눈으로 수집하였습니다.
- 데이터 수집 기준
  - reference image: 얼굴 윤곽이 뚜렷하게 보이는 정면 사진
  - 닮은꼴 찾기 이미지 : 얼굴 윤곽이 뚜렷하게 보이는 정면 사진, 화장이 잘 보이는 정면 사진
### 데이터 정제 
  - 1차 검수: Dlib 모델을 활용하여 얼굴 부분을 인식, 얼굴 기준으로 정렬(align) 하여 256 x 256 사이즈로 잘라주는 알고리즘을 사용하여 정제하였다.![crop & align code](https://github.com/junnnn-a/About_Me/tree/main/Data/crop%20%26%20align)
  - 2차 검수: 눈과 손으로 미러링, 정렬이 맞지 않는 사진을 제거하였음
  



## Stargan V2 모델 학습 데이터

### 데이터 수집
  - 수집 방법
  웹 크롤링 코드를 활용하여 수집하였음
  - 데이터 수집 조건 기준
    - 신체부위(팔, 손가락)이 얼굴에 닿으면 안됨
    - 안경, 마이크등을 착용하면 안됨
    - 정면을 바라보는 사진, 측면 사진 안됨
  - 중국인 생성 데이터를 활용하였음. Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
  
### 데이터 정제
- 1차 검수: 웹 크롤링 후 눈과 손으로 조건에 부합하는 사진 선별하였음
- 2차 검수: Dlib 모델을 활용하여 얼굴 부분을 인식, 얼굴 기준으로 정렬(align) 하여 256 x 256 사이즈로 잘라주는 알고리즘을 사용하여 정제하였다.![crop & align code](https://github.com/junnnn-a/About_Me/tree/main/Data/crop%20%26%20align)
- 3차 검수: 눈과 손으로 미러링, 정렬이 맞지 않는 사진을 제거하였음


