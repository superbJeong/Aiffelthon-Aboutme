# About me model
About me model은 Dlib 모델을 활용한 닮은꼴 찾기 알고리즘 코드와, Stargan-v2 모델로 구성되어 있습니다.
## 1. Dlib 모델을 활용한 닮은꼴 찾기 알고리즘
### 닮은꼴 찾기 알고리즘 코드 소개
- Dlib모델 데이터를 (연예인 얼굴) 29개의 조건으로 수치화
  - 28개 조건
    - 턱 각도 계산, 얼굴에서 코 비율, 얼굴에서 눈 비율, 중안부, 하안부 비율
- User_image를 입력시 Dlib모델로 29개 조건으로 수치화
  - 28개 조건
    - 턱 각도 계산, 얼굴에서 코 비율, 얼굴에서 눈 비율, 중안부, 하안부 비율
- 연예인의 데이터 수치와 비교하여 score 계산
- Score가 높은 3개 연예인 사진 선택

## 2. Stargan-v2 모델
![image](https://user-images.githubusercontent.com/97006756/172538439-de81d886-e0af-4084-8398-2117e0f7f17f.png)
![image](https://user-images.githubusercontent.com/97006756/172538468-a7a4cb95-b009-4fa3-8e59-f13c3253c642.png)

### stargan-v2 모델 선정 이유
- 한번에 여러 개의 도메인으로의 변환이 가능하기 때문(초반 목표로는 여자, 남자 메이크업, 헤어스타일 추천시스템도 만들어보기로 했기 때문)
- 사용자의 얼굴형과 이목구비를 잘 가져오는 좋은 성능 때문



