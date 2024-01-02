"""
爬取笔趣阁小说数据
"""
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
import pandas  as pd

host = "https://www.qidian.com/all"
headers = {
    "Host": "www.qidian.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}

res_header = ["类别", "标题", "简介"]
table = []
total = 0

classes = {
    "玄幻": {"base-url": host + "/chanId21", "num_page": 5},
    "奇幻": {"base-url": host + "/chanId1", "num_page": 5},
    "武侠": {"base-url": host + "/chanId2", "num_page": 5},
    "仙侠": {"base-url": host + "/chanId22", "num_page": 5},
    "都市": {"base-url": host + "/chanId4", "num_page": 5},
    "现实": {"base-url": host + "/chanId15", "num_page": 5},
    "军事": {"base-url": host + "/chanId6", "num_page": 5},
    "历史": {"base-url": host + "/chanId5", "num_page": 5},
    "游戏": {"base-url": host + "/chanId7", "num_page": 5},
    "体育": {"base-url": host + "/chanId8", "num_page": 5},
    "科幻": {"base-url": host + "/chanId9", "num_page": 5},
    "悬疑": {"base-url": host + "/chanId10", "num_page": 5},
    "古代言情": {"base-url": host + "/chanId80", "num_page": 5},    # 所有言情归为一类
    "现代言情": {"base-url": host + "/chanId82", "num_page": 5},
    "幻想言情": {"base-url": host + "/chanId83", "num_page": 5},
}

def get_target_url(base_url, page):
    return base_url + f"-page{page}/"


async def fetch_content(session, url):
    async with session.get(url, headers=headers) as response:
        return await response.text()


def parse_content(content) -> list:
    soup = BeautifulSoup(content, 'html.parser')

    # 找出所有具有属性data-rid的li标签
    books = soup.find_all('li', attrs={'data-rid': True})
    res = []
    # print(f"\n成功解析: {len(books)}")
    

    # 遍历每一个标签，提取信息
    for div in books:
        # 提取class为book-img-box的div标签下a标签的超链接
        href = div.find('div', class_="book-img-box").a.get('href')
        url = "https:" + href
        response = requests.get(url, headers=headers)
        
        # 提取class为book-mid-info的div标签
        sub_div = div.find('div', class_="book-mid-info")
        # 提取h2子标签的文本内容
        title = sub_div.h2.text.strip()

        # 提取class为author的p标签下第二个a标签的文本内容
        genre = sub_div.find('p', class_='author').find_all('a')[1].text.strip()    # 提取类别

        # 检查请求是否成功
        if response.status_code == 200:
            # 获取 HTML 内容
            sub_content = response.text
            sub_soup = BeautifulSoup(sub_content, 'html.parser')
            summary = sub_soup.find('p', id='book-intro-detail').get_text(separator=' ').replace('\n', ' ') # 提取简介并去除换行
        else:
            # 提取class为intro的p标签的文本内容
            summary = sub_div.find('p', class_='intro').text.strip()    # 用简短版本的简介代替
        
        
        row = [genre, title, summary]
        res.append(row)
        print(f"\r[INFO] 解析成功: 类别: {genre} 标题: {title} 简介: {summary[:20]}", end='')
    return res

async def main():
    global table
    async with aiohttp.ClientSession() as session:
        tasks = []

        for gener in classes.keys():
            num_pages = classes[gener]['num_page']
            for page in range(1, num_pages + 1):
                
                url = get_target_url(classes[gener]["base-url"], page)

                task = fetch_content(session, url)
                tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for content in responses:
            rows = parse_content(content)
            # new_row = {col: data for col, data in zip(res_header, parsed_data)}
            table += rows

if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n\n解析总数: {len(table)}")
    df = pd.DataFrame(table, columns=res_header)
    df.to_excel('output.xlsx', index=False)


