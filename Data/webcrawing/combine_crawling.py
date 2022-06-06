from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import urllib.request
import time, os


driver = webdriver.Chrome(r"D:\AIFFEL\webcrawling\chromedriver.exe")#chromedriver를 사용하기위한 webdriver함수 사용

keyword = str(input("insert keyword for searching : "))#get keyword to search

# 이미지 폴더 생성
if not os.path.isdir(keyword):
    os.makedirs(keyword)

driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")##open google image search page
# driver.maximize_window()##웹브라우저 창 화면 최대화
time.sleep(2)
driver.find_element_by_css_selector("input.gLFyf").send_keys(keyword) #send keyword
driver.find_element_by_css_selector("input.gLFyf").send_keys(Keys.RETURN)##send Keys.RETURN


last_height = driver.execute_script("return document.body.scrollHeight") #initialize standard of height first

while True: #break가 일어날 때 까지 계속 반복
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #페이지 스크롤 시키기

    time.sleep(1)

    new_height = driver.execute_script("return document.body.scrollHeight") ## update new_height
    if new_height == last_height:#이전 스크롤 길이와 현재의 스크롤 길이를 비교
        try:
            driver.find_element_by_css_selector(".mye4qd").click() ## click more button 더보기 버튼이 있을 경우 클릭
        except:
            break # 더보기 버튼이 없을 경우는 더 이상 나올 정보가 없다는 의미이므로 반복문을 break
    last_height = new_height ##last_height update

i=0

list = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")##thumnails list
print(len(list)) #print number of thumnails


for img in list:
    webdriver.ActionChains(driver).click(img).perform()
    driver.implicitly_wait(10)  # 로딩이 될 때 까지 기둘링 플리즈
    time.sleep(2)
    imgurl = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
    try:
        urllib.request.urlretrieve(imgurl, str(keyword) + '/' + str(keyword)+str(i)+".jpg")
        i+=1
    except:
        pass

driver.close()