import os
import sys
import cv2
import numpy as np
import dlib
from utils.wing_ import FaceAligner
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

def cropping(img_org, img_size, model_path):
  landmark_predictor = dlib.shape_predictor(model_path)
  detector_hog = dlib.get_frontal_face_detector()

  img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
  dlib_rects = detector_hog(img_rgb, 1)

  if not dlib_rects:
    print('얼굴 영역을 찾지 못했습니다.')
    return np.zeros([0, 0, 0])

  for j, dlib_rect in enumerate(dlib_rects):
    if j == 1:
      break

    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()
    w = r-l
    h = b-t
    new_l = l - int(w*0.5) if l - int(w*0.5) > 0 else 0
    new_t = t - int(h*0.75) if t - int(h*0.75) > 0 else 0
    new_r = r + int(w*0.5) if r + int(w*0.5) < img_org.shape[1] - 1 else img_org.shape[1] - 1
    new_b = b + int(h*0.25) if b + int(h*0.25) < img_org.shape[0] -1 else img_org.shape[0] -1
    new_w = (new_r) - (new_l)
    new_h = (new_b) - (new_t)

    # 수정된 부분 -> width와 height가 같도록 조절해줌
    if new_w > new_h:
      diff_wh = (new_w - new_h) // 2
      new_l = new_l + diff_wh
      new_r = new_r - diff_wh
      new_w = new_r - new_l
    elif new_w < new_h:
      diff_wh = (new_h - new_w) // 2
      new_t = new_t + diff_wh
      new_b = new_b - diff_wh
      new_h = new_b - new_t
    
    if new_w <= img_size or new_h <= img_size:
      print('얼굴 영역의 크기가 작습니다.')
      return np.zeros([0, 0, 0])
    img_cropped = img_org[ new_t : new_b, new_l : new_r ]

    points = landmark_predictor(img_rgb, dlib_rect)
    # face landmark 좌표를 저장해둡니다
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    if len(list_points) != 68:
      print('얼굴의 구성 요소가 잘 보이지 않습니다. (눈, 눈썹, 코, 입, 얼굴 윤곽 등)')
      return np.zeros([0, 0, 0])
    
  return img_cropped

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def align(aligner, transform, img_cropped):
  img_cropped_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
  x = transform(img_cropped_pil)
  x_aligned = aligner.align(x.unsqueeze(0))
  return x_aligned

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def HangulFormat(file_path):
  f = open(file_path.encode("UTF-8"), "rb")
  bytes = bytearray(f.read())
  npArr = np.asarray(bytes, dtype=np.uint8)

  return cv2.imdecode(npArr, cv2.IMREAD_UNCHANGED)

def crop_align():

  # 결과 이미지 사이즈
  img_size = 256

  # 사용자 이미지 디렉토리 확인
  dir_img = 'inputs/usr'
  if not os.path.isdir(dir_img):
    sys.exit('경로 설정이 올바르지 않습니다')
  file_list = os.listdir(dir_img)
  
  # src 디렉토리 확인
  dir_cropped = 'inputs/src/female'
  if not os.path.isdir(dir_cropped):
    os.mkdir(dir_cropped)

  # dlib 모델 디렉토리 설정  
  model_path = 'models/shape_predictor_68_face_landmarks.dat'
  
  # align 모델 디렉토리 설정
  wing_path = 'models/wing.ckpt'
  lm_path = 'models/celeba_lm_mean.npz'

  # align 모듈 불러오기
  aligner = FaceAligner(wing_path, lm_path, img_size)
  transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


  for img_file in file_list:
    
    path = dir_img + '/' + img_file
    

    img_org = HangulFormat(path)      
    img_cropped = cropping(img_org, img_size, model_path)

    if not img_cropped.any():
      continue 

    x_aligned = align(aligner, transform, img_cropped)
    # img_cropped_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    # x = transform(img_cropped_pil)
    # x_aligned = aligner.align(x.unsqueeze(0))

    filename = os.path.join(dir_cropped, '_'.join(img_file.split(' '))[:-3]+'jpg')
    save_image(x_aligned, 1, filename=filename)