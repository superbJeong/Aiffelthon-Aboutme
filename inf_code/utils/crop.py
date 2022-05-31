'''
[220513]

* 기본적으로 StarGan-v2의 PyTorch 버젼이 돌아가는 환경이면 실행 가능

* crop_align.py 파일이 있는 폴더에 images 폴더를 만들고 그 안에 pre 폴더를 만들어서 원본 이미지를 넣어놔야함
  결과는 images/processed 폴더에 저장됨

* dlib 라이브러리를 설치해야 함
    -> 아나콘다 가상환경이면 터미널에 conda install -c conda-forge dlib 입력하여 설치

* __main__ 내부에서 폴더 경로를 바꿔도 되고,
  터미널에서 실행시 아래 명령어의 'pre'와 'processed' 부분을 변경하여 폴더 지정 및 실행 가능
    -> python crop_align_v1.py --inp_dir images/pre --out_dir images/processed
'''

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from wing_ import align_faces, FaceAligner
import argparse
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

def cropping(i, img_org, img_size = 240):
  try:
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    dlib_rects = detector_hog(img_rgb, 1)
    wi_he = []

    if not dlib_rects:
      print('얼굴 영역 못찾음')
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
        print(f'{i}번 째 이미지의 {len(dlib_rects)}명 중 {j+1}번 째 얼굴 영역 크기가 작음')
        return np.zeros([0, 0, 0])
      img_cropped = img_org[ new_t : new_b, new_l : new_r ]

      points = landmark_predictor(img_rgb, dlib_rect)
      # face landmark 좌표를 저장해둡니다
      list_points = list(map(lambda p: (p.x, p.y), points.parts()))
      if len(list_points) != 68:
        print(f'{i}번 째 이미지의 {len(dlib_rects)}명 중 {j+1}번 째 얼굴 랜드마크가 온전치 않음')
        return np.zeros([0, 0, 0])
      
    return img_cropped


  except(AttributeError):
      print(f'ERROR: {i}번 째 이미지 에러로 인한 예외 처리')
      print('------------------------------------------------------------------')
      return np.zeros([0, 0, 0])


def bgrshow(img):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def HangulFormat(file_path):
  f = open(file_path.encode("UTF-8"), "rb")
  bytes = bytearray(f.read())
  npArr = np.asarray(bytes, dtype=np.uint8)

  return cv2.imdecode(npArr, cv2.IMREAD_UNCHANGED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # model arguments
  parser.add_argument('--img_size', type=int, default=256,
                          help='Image resolution')

  # face alignment
  parser.add_argument('--wing_path', type=str, 
                          default='needs/wing.ckpt')
  parser.add_argument('--lm_path', type=str, 
                          default='needs/celeba_lm_mean.npz')

  # directory for testing
  parser.add_argument('--inp_dir', type=str, default='images/pre', 
                          help='input directory when aligning faces')
  parser.add_argument('--out_dir', type=str, default='images/processed', 
                          help='output directory when aligning faces')

  args = parser.parse_args()

  dir_img = args.inp_dir
  if not os.path.isdir(dir_img):
    sys.exit('경로 설정이 올바르지 않습니다')
  file_list = sorted(os.listdir(dir_img))
  model_path = 'needs/shape_predictor_68_face_landmarks.dat'
  landmark_predictor = dlib.shape_predictor(model_path)
  detector_hog = dlib.get_frontal_face_detector()

  dir_cropped = args.out_dir
  if not os.path.isdir(dir_cropped):
    os.mkdir(dir_cropped)


  cnt = 0
  for i, img_file in enumerate(file_list):
    aligner = FaceAligner(args.wing_path, args.lm_path, args.img_size)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    print('------------------------------------------------------------------')
    print(f'{i}번 째 이미지 \"{img_file}\" 처리시작')
    path = dir_img + '/' + img_file
    
    try:
      img_org = HangulFormat(path)      
      img_cropped = cropping(i, img_org, args.img_size)

      if not img_cropped.any():
        continue 

      img_cropped_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
      x = transform(img_cropped_pil)
      x_aligned = aligner.align(x.unsqueeze(0))

      filename = os.path.join(dir_cropped, '_'.join(img_file.split(' '))[:-3]+'jpg')
      # filename = dir_cropped + '/' + '_'.join(img_file.split(' '))[:-3] + 'jpg' 
      save_image(x_aligned, 1, filename=filename)

      # print(filename)

      # if i == 0:
      #   break

      cnt += 1
      print(f'{i}번 째 이미지 완료 {filename}')
      print('------------------------------------------------------------------')
    except:
      print(f'{i}번 째 이미지 \"{img_file}\" 잘못 됨')
      print('------------------------------------------------------------------')
  
  print(f'총 {cnt}개 이미지 저장 완료')