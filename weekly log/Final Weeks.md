# **8주차 ( 06/06 ~ 06/09 )**

## **주요 내용**
  - 모델 하이퍼 파라미터 선정, 입력 이미지와 닮은꼴 연예인 매칭 
  - 웹페이지에 모델 얹고 시현
  - 프로젝트 정리, github 마무리 


<br/>



----------

## **입력 이미지와 닮은 꼴의 연예인 이미지 매칭**
- 입력 이미지의 스코어를 산출하여, 닮은꼴 비교용 스코어와 비교한 다음, 일치하는 연예인을 찾아, 그 연예인의 화장 이미지를 레퍼런스 이미지로 선정 하는 과정
- 위 과정을 코드로 경민님께서 구현


<br/>



## **레퍼런스 이미지 선별**
- 닮은꼴 비교용 연예인 사진을 연예인 1명당 5장, 레퍼런스 이미지용 1장을 찾아 데이터 라벨링 작업
- 닮은꼴 비교용 연예인 사진에 대한 스코어 산출
- 6-7주차에 수집한 연예인의 이름과 스코어를 dataset으로 저장
    

<br/>



## **발표 준비**
- 팀원 소개, 모델 소개 및 결과 ppt로 간략 정리
- 세부 내용은 github readme에 분야별로 기술