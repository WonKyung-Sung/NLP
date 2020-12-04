# -*- coding: utf-8 -*-
#import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import numpy as np
import time
'''
part1
'''

# search = open("C:/Users/W.K.SUNG/Desktop/Bin/naver2014_05_12.txt",'w')
chrome_path = r"/data3/home/wonkyung.sung/Crome_driver/chromedriver"
driver = webdriver.Chrome(chrome_path)
df = pd.DataFrame()
import urllib
query = ["세탁기", "노트북", "건조기", "티비", "스마트폰", "모니터"]

# query = ["삼성"]
query_ = [urllib.parse.quote(i) for i in query]


for query_first in query:
    print(query_first, " 시작")

    driver.get("https://shopping.naver.com/")
    driver.find_element_by_xpath("""//*[@id="autocompleteWrapper"]/input[1]""").send_keys(query_first)
    driver.find_element_by_xpath("""//*[@id="autocompleteWrapper"]/a[2]""").click()

    tmp_df = pd.DataFrame()

    # 크롤링
    time.sleep(5)
    driver.find_element_by_xpath("""//*[@id="__next"]/div/div[2]/div/div/div[1]/div[1]/ul/li[2]""").click()


    next_page_ =True 
    count = 0 
         
    while next_page_:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # e = driver.find_elements_by_xpath("""//*[@id="__next"]/div/div[2]/div/div[4]/div[1]/ul/div/div""")
        try:
            e = driver.find_element_by_class_name("list_basis").find_elements_by_tag_name("li")
        except:
            continue
        #print(str(len(e)), "개를 긁었습니다.")
        # count = 0 
        count_o = 0 
        for th in e:
            try:
                th.find_element_by_class_name("basicList_etc_box__1Jzg6")

                raw = th.text

                # 제품 카테고리
                product_lines = th.find_element_by_class_name("basicList_depth__2QIie").find_elements_by_tag_name("a")
                product_line = ";;".join([product_line.text for product_line in product_lines])

                # 제품명
                product_name = th.find_element_by_class_name("basicList_title__3P9Q7").find_element_by_tag_name("a").text

                # 가격
                # product_name = th.find_element_by_class_name("basicList_price_2r23")

                price_infos = th.find_element_by_class_name("basicList_price__2r23_")
                price = price_infos.find_element_by_class_name("price_num__2WUXn").text
                number_of_store = price_infos.find_element_by_tag_name("a").text.replace("판매처 ", "")
                detailes = th.find_elements_by_class_name("basicList_detail__27Krk")
                detailes = ";;".join([tmp.text for tmp in detailes])
                review_info = th.find_element_by_class_name("basicList_etc_box__1Jzg6")
                review_url = review_info.find_element_by_tag_name("a").get_attribute('href')
                review_star = review_info.find_element_by_class_name("basicList_star__3NkBn").text.replace("별점 ", "")
                review_number_of = review_info.find_element_by_class_name("basicList_num__1yXM9").text

                tmp_df = tmp_df.append(pd.DataFrame({"product_name":[product_name],
                                                     "검색명" : [query_first],
                                                     "제품카테고리":[product_line],
                                                     "price": [price],
                                                     "판매처": [number_of_store],
                                                     "detailes": [detailes],
                                                     "별점":[review_star],
                                                     "review_url":[review_url],
                                                     "리뷰수":[review_number_of],
                                                     "raw":[raw]}), ignore_index=True)

                # count_o += 1 
                # print("   ", str(count_o))

            except:
                # count+= 1
                # print(count)
                pass
        df = df.append(tmp_df, ignore_index=True)
        
        df = df.drop_duplicates()

        next_page_sum = ""
        for next_page_tmp in driver.find_element_by_xpath("""//*[@id="__next"]/div/div[2]/div[2]/div[4]/div[1]/div[3]""").find_elements_by_tag_name("a"):
            # //*[@id="__next"]/div/div[2]/div[2]/div[3]/div[1]/div[3] # 이거 
            # //*[@id="__next"]/div/div[2]/div/div[3]/div[1]/div[3]
            if next_page_tmp.text == "다음":
                next_page_sum = next_page_sum + "다음"
                next_page_tmp.send_keys(Keys.ENTER)
                count+= 1
                print(count)
                if (query_first is not ["삼성", "엘지"])& (count > 300):
                    print(1)
                    next_page_ = False
        if "다음" not in next_page_sum:
            next_page_ = False
    df = df.drop_duplicates()
    df.drop_duplicates().to_csv("제품명_1101_" + query_first +".csv", index=False)
