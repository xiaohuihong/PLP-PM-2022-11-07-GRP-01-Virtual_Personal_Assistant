from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os


new_num = 10
# &category=business,entertainment,politics,sports,tech
res = requests.get("https://search.api.cnn.io/content?q=news&size=%s&sort=newest&from=0"%new_num)
links = [x['url'] for x in res.json()['result']]



url_list = []
title_list = []
content_list = []

for url in links:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # getting the title element
    if soup.find('h1'):
        title = re.sub(r"\n[ ]*", "", soup.find('h1').getText())
    else:
        continue

    # # getting author + update and separate
    # author_and_update = soup.find(class_="Article__subtitle").get_text()
    # splited = author_and_update.split('CNN •')
    #
    # # the author data
    # author = re.sub(",", '', splited[0])

    # the update data
    # update_date = re.sub("Updated", "", splited[1])
    # update_date = re.sub("^\s+", '', update_date)

    # the text content
    content_text = []
    for text in soup.findAll(class_=['paragraph', 'Paragraph__component']):
        content_text.append(re.sub(r"\n[ ]*", "", text.get_text()))
    content = ''.join(content_text)

    # Bonus => image content
    # images = []
    # for image in soup.find_all("img"):
    #     images.append(image.get('src'))

    # # image treatment => No Blur + Size
    # cleaned_img = []
    # for link in images:
    #     link = re.sub("e_blur:500", "e_blur:0", link)
    #     link = re.sub("w_50", "w_634", link)
    #     link = re.sub("h_28", "h_357", link)
    #     cleaned_img.append(link)



    # dictionnary to structure data
    # scraped_data = {
    #     "url" : url,
    #     "title" : title,
    #     # "author" : author,
    #     # "date_update": update_date,
    #     # "image": cleaned_img[0],
    #     "content": content
    # }

    url_list.append(url)
    title_list.append(title)
    content_list.append(content)

data = {'url': url_list,
        'title': title_list,
        'content': content_list
        }

df = pd.DataFrame(data, columns=['url', 'title', 'content'])

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#write structure data to excel
df.to_csv("./today_news.csv", sep='|', encoding='utf-8', index=False)
