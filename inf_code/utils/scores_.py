import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import glob
import sys



# 디렉토리 설정
ref_path = 'inputs/reference'
usr_path = 'inputs/src/female'

dlib_model_path = 'models/shape_predictor_68_face_landmarks.dat'


# 이미지 경로 불러오기
images = glob.glob (ref_path + '\\*')
if not images:
  sys.exit()
print(f'{len(images)} images path load complete!')

# face 및 landmark detector 선언
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(dlib_model_path)

# 얼굴 및 랜드마크 구하는 함수들 정의

################################################################################
'''
얼굴 찾는 함수
이미지에 한 사람만 있다고 가정
한명의 face 좌표 리스트를 return
'''
def get_face(img):
  # face_list = []
  det = detector(img)[0]
  
  # # 얼굴 출력해보기
  # fig, ax = plt.subplots(1, figsize=(5, 5))
  # img_result = img.copy()
  # x, y, w, h = det.left(), det.top(), det.width(), det.height()
  # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
  # ax.add_patch(rect)  
  # ax.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
  # # 여기까지

  return det
################################################################################
'''
얼굴에서 랜드마크 찾는 함수
이미지와 face 좌표를 입력하여 landmark를 return
'''
def get_landmark(img, face):
  points = sp(img, face)
  list_points = list(map(lambda p: (p.x, p.y), points.parts()))

  # # 출력해보기
  # img_result = img.copy()
  # fig, ax = plt.subplots(1, figsize=(5, 5))
  # for point in points.parts():
  #   circle = patches.Circle((point.x, point.y), radius=2, edgecolor='r', facecolor='r')
  #   ax.add_patch(circle)
  # ax.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
  # # 여기까지

  return list_points
  ################################################################################

################################################################################
## 점 3개 잡아서 angle 구하는 함수 (반환값: 각도)
def angle_eyes(topPoint, midPoint, botPoint):
  # print(topPoint, midPoint, botPoint)
  
    ## midPoint와 topPoint 의 x 좌표 값이 같은 경우
  if (midPoint[0] - topPoint[0]) == 0 and (midPoint[0] - botPoint[0]) != 0:
    x =  abs((midPoint[1] - botPoint[1]) / (midPoint[0] - botPoint[0]))
    angle_rad = np.arctan(x)
    angle_deg_1 = np.degrees(angle_rad)
    angle_result = 180 - angle_deg_1
    # print("here")
    return angle_result

  ## midPoint와 botPoint 의 x 좌표 값이 같은 경우
  elif (midPoint[0] - botPoint[0]) == 0 and (midPoint[0] - topPoint[0]) != 0:
    x = abs((midPoint[1] - topPoint[1]) / (midPoint[0] - topPoint[0]))
    angle_rad = np.arctan(x)
    angle_deg_1 = np.degrees(angle_rad)
    angle_result = 180 - angle_deg_1
    # print("hereeeee")
    return angle_result

  elif (midPoint[0] - botPoint[0]) == 0 and (midPoint[0] - topPoint[0]) == 0:
    # print("hhhhhhhhere")
    return 0

  else:
    topLineSlope = (midPoint[1] - topPoint[1]) / (midPoint[0] - topPoint[0]) #기울기1 구하기
    botLineSlope = (midPoint[1] - botPoint[1]) / (midPoint[0] - botPoint[0]) #기울기2 구하기
    x = (abs(topLineSlope - botLineSlope)) / (1 + topLineSlope * botLineSlope) #x 구하기
    angle_rad = np.arctan(x) #x의 arctan값 구하기 / 구하면 radian값이 나옵니다.
    angle_deg = np.degrees(angle_rad) #radian값을 degree값으로 바꿔줍니다.
    if angle_deg < 90:
      angle_deg = 180 - angle_deg
    if topPoint[1] < midPoint[1]: # 왼쪽 눈 끝이 교차점 보다 아래에 있으면 쳐진 눈(스코어 > 0.5)
      angle_deg = 360 - angle_deg
    # print(angle_rad, angle_deg)
    return angle_deg
################################################################################
## 왼쪽,오른쪽 눈매 기울기 각도 (값 1개)
def shape_of_eyes_ratio_(lm):
  cp = get_crosspt(lm[36], lm[39], lm[42], lm[45])
  if cp == None:
    face_list = [0.5]
    return face_list
  cp = list(map(round, cp))
  # print(lm[36], lm[39], cp, lm[42], lm[45])
  degree = angle_eyes(lm[36], cp, lm[45])
  face_list = [str(round(degree/360, 3))]
  # print(degree, face_list[0])
  return face_list
################################################################################
## 왼쪽,오른쪽 눈매 기울기 (값 2개)
def shape_of_eyes_ratio(lm):
  l_shape_eyes = (lm[39][1]-lm[36][1]/lm[39][0]-lm[36][0])
  r_shape_eyes = (lm[42][1]-lm[45][1]/lm[42][0]-lm[45][0])
  face_list = [str(round(l_shape_eyes, 3)), str(round(r_shape_eyes, 3))]
  return face_list
################################################################################
## 전체 얼굴 비율 (값 1개)
def full_face_ratio(lm):
  face_length = (lm[8][1] - lm[22][1])
  full_face_length = face_length * 1.3
  width_1 = abs(lm[0][0]-lm[16][0])
  width_2 = abs(lm[1][0]-lm[15][0])
  width_3 = abs(lm[2][0]-lm[14][0])

  full_face_width = max(width_1, width_2, width_3)
  full_face = full_face_width / full_face_length
  face_list = [str(round(full_face, 3))]
  return face_list
################################################################################

# 비율로 score 구하는 함수들 한번에 정의

################################################################################
## 점 3개 잡아서 angle 구하는 함수 (반환값: 각도)
def angle(topPoint, midPoint, botPoint):
  
  ## midPoint와 topPoint 의 x 좌표 값이 같은 경우
  if (midPoint[0] - topPoint[0]) == 0 and (midPoint[0] - botPoint[0]) != 0:
    x =  abs((midPoint[1] - botPoint[1]) / (midPoint[0] - botPoint[0]))
    angle_rad = np.arctan(x)
    angle_deg_1 = np.degrees(angle_rad)
    angle_result = 90 + angle_deg_1
    normalization_angle = (angle_result - 90) / 90
    return angle_result

  ## midPoint와 botPoint 의 x 좌표 값이 같은 경우
  elif (midPoint[0] - botPoint[0]) == 0 and (midPoint[0] - topPoint[0]) != 0:
    x = abs((midPoint[1] - topPoint[1]) / (midPoint[0] - topPoint[0]))
    angle_rad = np.arctan(x)
    angle_deg_1 = np.degrees(angle_rad)
    angle_result = 90 + angle_deg_1
    normalization_angle = (angle_result - 90) / 90
    return angle_result

  elif (midPoint[0] - botPoint[0]) == 0 and (midPoint[0] - topPoint[0]) == 0:
    return 180

  ## 그 외 일반적인 좌표인 경우
  else:
    topLineSlope = (midPoint[1] - topPoint[1]) / (midPoint[0] - topPoint[0]) #기울기1 구하기
    botLineSlope = (midPoint[1] - botPoint[1]) / (midPoint[0] - botPoint[0]) #기울기2 구하기
    x = (abs(topLineSlope - botLineSlope)) / (1 + topLineSlope * botLineSlope) #x 구하기
    angle_rad = np.arctan(x) #x의 arctan값 구하기 / 구하면 radian값이 나옵니다.
    angle_deg = np.degrees(angle_rad) #radian값을 degree값으로 바꿔줍니다.
    if angle_deg < 90: #교차각이 2개 나오므로, 90도보다 작은 각이 나오면 180도에서 이를 뺀 각을 교차각 최종값으로 정의합니다.
      angle_result = 180 - angle_deg
      normalization_angle = (angle_result - 90) / 90
    else:
      angle_result = angle_deg
      normalization_angle = (angle_result - 90) / 90
    return angle_result #함수의 리턴값을 정의합니다.
################################################################################
# lm: landmark를 의미
################################################################################
# 각도들을 radian 값으로 바꿔서 하나의 리스트로 반환해주는 함수 (값 15개)
def avg_angle_chin_and_clown(lm):
  angle_result = []
  for i in range(0, 15):
    angle_degree = angle(lm[i], lm[i+1], lm[i+2])
    angle_result.append(str(round(angle_degree/180, 3)))
  return angle_result
################################################################################
## 중안부 하안부 비율 (값 2개)
def middle_part_lower_part_ratio(lm):
  a= (lm[8][1]-lm[22][1])
  b= (lm[33][1]-lm[22][1])
  c= (lm[8][1]- lm[33][1])

  middle = ((b/a))
  bottom = ((c/a))
  face_list = [str(round(middle,3)), str(round(bottom,3))]
  return face_list
################################################################################
## 볼과 입술의 비율 (값 1개)
def cheeks_and_lips_ratio(lm):
  cheek = (lm[12][0]-lm[4][0])
  lip = (lm[54][0]-lm[48][0])

  cheek_lip = (lip/cheek)
  face_list = [str(round(cheek_lip,3))]
  return face_list
################################################################################
## 코와 얼굴너비 비율 (값 1개)
def face_and_nose_ratio(lm):
  face = (lm[14][0]- lm[2][0])
  nose = (lm[35][0]-lm[31][0])

  face_nose = (nose/face)
  face_list = [str(round(face_nose,3))]
  return face_list
################################################################################
## 미간과 눈 길이 (값 3개)
def betweend_the_eyebrows_ratio(lm):
  eyes_1 = abs(lm[36][0]- lm[39][0])
  eyes_2 = abs(lm[39][0]- lm[42][0])
  eyes_3 = abs(lm[42][0]- lm[45][0])
  face_eyes = abs(lm[0][0]-lm[16][0])
  eyes_eyes = abs(lm[36][0]-lm[45][0])

  eyes_left = eyes_1/face_eyes
  between_the_eyebrows = eyes_2/eyes_eyes
  eyes_right = eyes_3/face_eyes
  face_list = [str(round(eyes_left, 3)), str(round(between_the_eyebrows, 3)), str(round(eyes_right, 3))]
  return face_list
################################################################################
## 눈 둘레와 너비의 비율 (값 2개)
def eyes_length_and_width(lm):
  length_left = 0
  for i in range(36, 41):
    length_temp = np.hypot(lm[i][0]-lm[i+1][0], lm[i][1]-lm[i+1][1])
    length_left += length_temp
  width_l = np.hypot(lm[36][0]-lm[39][0], lm[36][1]-lm[39][1])

  length_right = 0
  for i in range(42, 47):
    length_temp = np.hypot(lm[i][0]-lm[i+1][0], lm[i][1]-lm[i+1][1])
    length_right += length_temp
  width_r = np.hypot(lm[42][0]-lm[45][0], lm[42][1]-lm[45][1])

  ratio_left = width_l/length_left
  ratio_right = width_r/length_right
  face_list = [str(round(ratio_left, 3)), str(round(ratio_right, 3))]
  return face_list
################################################################################
## 눈 둘레와 너비의 비율 (값 2개)
def eyes_length_and_height(lm):
  length_left = 0
  for i in range(36, 41):
    length_temp = np.hypot(lm[i][0]-lm[i+1][0], lm[i][1]-lm[i+1][1])
    length_left += length_temp
  height_l1 = np.hypot(lm[37][0]-lm[41][0], lm[37][1]-lm[41][1])
  height_l2 = np.hypot(lm[38][0]-lm[40][0], lm[38][1]-lm[40][1])
  height_l = (height_l1+height_l2) / 2

  length_right = 0
  for i in range(42, 47):
    length_temp = np.hypot(lm[i][0]-lm[i+1][0], lm[i][1]-lm[i+1][1])
    length_right += length_temp
  height_r1 = np.hypot(lm[43][0]-lm[47][0], lm[43][1]-lm[47][1])
  height_r2 = np.hypot(lm[44][0]-lm[46][0], lm[44][1]-lm[46][1])
  height_r = (height_r1+height_r2) / 2

  ratio_left = height_l/length_left
  ratio_right = height_r/length_right
  face_list = [str(round(ratio_left, 3)), str(round(ratio_right, 3))]
  return face_list
################################################################################
## 하관 너비 비율 (값 1개)
def bottom_width(lm):
  width_1 = abs(lm[0][0]-lm[16][0])
  width_2 = abs(lm[1][0]-lm[15][0])
  width_3 = abs(lm[2][0]-lm[14][0])
  width_b = abs(lm[5][0]-lm[11][0])

  width_max = max(width_1, width_2, width_3)
  ratio_width = width_b / width_max
  face_list = [str(round(ratio_width, 3))]
  return face_list
################################################################################
## 하관 높이 비율 (값 1개)
def bottom_heiht(lm):
  height_b = abs(lm[57][1]-lm[8][1])
  height_comp = abs(lm[33][1]-lm[8][1])

  ratio_height = height_b / height_comp
  face_list = [str(round(ratio_height, 3))]
  return face_list
################################################################################
## score들을 구하는 함수
def get_scores_list(lm):
  face_list = []
  face_list += avg_angle_chin_and_clown(lm)
  face_list += middle_part_lower_part_ratio(lm)
  face_list += cheeks_and_lips_ratio(lm)
  face_list += face_and_nose_ratio(lm)
  face_list += betweend_the_eyebrows_ratio(lm)
  face_list += eyes_length_and_height(lm)
  face_list += bottom_width(lm)
  face_list += bottom_heiht(lm)
  face_list += shape_of_eyes_ratio_(lm)
  face_list += full_face_ratio(lm)
  return face_list
################################################################################
def get_scores_dict(lm):
  face_list = {}
  face_list["angles"]= avg_angle_chin_and_clown(lm)
  face_list["mid_low_ratio"]= middle_part_lower_part_ratio(lm)
  face_list["cheek_lip_ratio"]=cheeks_and_lips_ratio(lm)
  face_list["face_nose_ratio"]=face_and_nose_ratio(lm)
  face_list["eyebrows_ratio"]=betweend_the_eyebrows_ratio(lm)
  face_list["eye_length_height_ratio"]=eyes_length_and_height(lm)
  face_list["bot_width_ratio"]=bottom_width(lm)
  face_list["bot_height_ratio"]=bottom_heiht(lm)
  face_list["shape_of_eyes_ratio"]=shape_of_eyes_ratio(lm)
  face_list["full_face_ratio"]=full_face_ratio(lm)
  return face_list

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

## 기존 score 불러오는 함수
def calulate_scores(usr_score):
  scores_path = 'inputs/scores_ref.txt'
  final_scores = {}
  with open(scores_path, 'r') as file:
    for i in range(25557):
      name = file.readline()
      scr = file.readline()
      # score = list(map(float,list(scr.split(','))))
      score = list(scr.split(','))
      # print(name, score, type(score))
      scores_tmp = 0
      for i in range(28):
        scores = abs(float(usr_score[i]) - float(score[i]))
        #print(scores)
        scores_tmp += scores
      #print('middle_scores',middle_scores)
      final_scores[name] = scores_tmp/28
      #print('final_scores',final_scores)
  return final_scores

## 가장 높은 score인 이미지 파일의 이름을 반환
def select_ref(final_scores):
  max_score = 0
  name = ''
  for key, value in final_scores.items():
    if value > max_score:
      name = key
      max_score = value
  return name

## score가 높은 3명의 이미지 파일 이름을 반환
def select_refs(final_scores):
  sorted_scores = sorted(final_scores.items(), key=lambda x:x[1], reverse=True)
  return sorted_scores[:3]


## 눈매 교차점 찾기
def get_crosspt(l_l, l_r, r_l, r_r):
  x11, y11 = l_l
  x12, y12 = l_r
  x21, y21 = r_l
  x22, y22 = r_r
  if x12==x11 or x22==x21:
    print('delta x=0')
    if x12==x11:
      cx = x12
      m2 = (y22 - y21) / (x22 - x21)
      cy = m2 * (cx - x21) + y21
      crosspoint = [cx, cy]
      return crosspoint
    if x22==x21:
      cx = x22
      m1 = (y12 - y11) / (x12 - x11)
      cy = m1 * (cx - x11) + y11
      crosspoint = [cx, cy]
      return crosspoint
    
  m1 = (y12 - y11) / (x12 - x11)
  m2 = (y22 - y21) / (x22 - x21)
  if m1==m2:
    print('parallel')
    return None
  print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
  cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
  cy = m1 * (cx - x11) + y11
  crosspoint = [cx, cy]
  return crosspoint
