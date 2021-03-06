# 2주차 ( 04/25 ~ 04/29 )

## 주요 내용 
  - 핵심 모델 검색, 닮은꼴 연예인 찾기 코드 구상 

<br/>

----------


## 데이터 수집
- 주말동안 수집하면서 느낀점, 수집 시 고려할 점 공유
- 모델 학습에 필요할 이미지 수집 기준 수립
    - 수집할 인물
        - 배우 → 얼굴형이 다양하고, 특징이 다 달라 다양한 얼굴형을 모델이 학습 할 수 있다. 소속사 홈페이지를 통해 프로필 사진 수집.
        - 아이돌 → 비슷한 사람이 많아 학습 데이터로는 부적격
    - 이미지 크기, 얼굴의 방향, 화장의 진함/연함 등


<br/>



## 화장 , 메이크업 적용 모델 찾기
- Beauty GAN
    - pretrained model 코드로 모델 구현 후 결과 값 확인
    - 화장은 대체적으로 잘 가져오나, 눈썹 부분을 잘 가져오지 못함
    - 얼굴이 살짝 기울어도 수평을 맞춰주는 얼라인 코드가 모델 내 포함되어 있었음
- StarGAN
    - 나동빈님 유투브 - 코드 분석 및 결과값을 확인 할 수 있는 영상 리뷰 확인
        
- StarGAN V2
    - tensorflow 로 구현된 모델은 사전학습된 모델이 없었다.
    - 학습되지 않은 모델로 주어진 데이터셋을 이용해 학습해 본 결과 reference를 제대로 가져오지 못함. 미흡한 점이 많았음.
    - pytorch로 구현된 모델에는 사전학습된 모델은 있었으나, 한번도 해 보지않아 어려움 봉착.
        - StarGAN V2 외에도 다른 모델들도 Pytorch로 구현된 모델이 많았음.
        - tensorflow 형식으로 바꾸는 방법도 있긴 했지만, 추후 다양한 파라미터 조정을 위해, pytorch의 사용 방법을 익혀야 할 것 같다.
  
- Beauty Glow
    - 모델 코드는 있으나, 구현이 어려워 포기
- Facial Attribute Transformers for Precise and Robust Makeup Transfer
- PSGAN++


<br/>



## `얼굴이 닮았다` 의 기준
- 얼굴의 어떤 부분을 어떤 식으로 보고 닮았다 라는 것을 찾아주기 위해 landmark 간의 거리를 산출해 score를 매겨 입력 이미지와 비교해 보기로 결정
- 턱 선 라인, 볼과 입술의 비율, 코 너비와 얼굴 너비의 비율 등.
    

<br/>



## 이전 기수 닮은 꼴 연예인 찾기 프로젝트
- 프로젝트 구현 후 결과 값 확인 → 입력이미지와 닮은 표정 찾기 코드였음.
- 비슷한 표정, 비슷한 조명인 사진을 넣으면 그런 사진을 찾아주는 형식
- 이번 프로젝트에선 활용할 수 없을 것같다 판단.