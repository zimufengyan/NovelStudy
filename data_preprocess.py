"""
处理原始数据, 按照8:1:1的比例划分为训练集、验证集和测试集
"""
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib
import unicodedata as ucd
import matplotlib.pyplot as plt


def process_text(text):
    text = text.strip()
    text = ucd.normalize('NFKC', text).replace(" ", "")
    text = text.replace("\n", "").replace("……", "").replace("......", "")
    return text


# 定义一个函数，用于读取一个excel文件并进行数据清洗
def read_excel(file):
    df = pd.read_excel(file)
    df = df.astype(str)
    # print(df.info())
    # 剔除简介字段只有数字的数据
    df = df[~df['简介'].str.isdigit()]
    # 剔除简介内容长度低于5的数据
    df = df[(df['简介'].str.len() >= 10) & (df['简介'].str.len() < 500)]
    df['简介'] = df['简介'].apply(lambda x: process_text(x))
    df['类别'] = df['类别'].apply(lambda x: process_text(x))
    return df


data_dir = "." + os.sep + "data" + os.sep + "original" + os.sep
output_dir = "." + os.sep + "data" + os.sep

# 定义一个空的DataFrame，用于存储所有读取的数据
all_data = pd.DataFrame()

# 遍历文件夹中的所有excel文件，并读取数据
for file in os.listdir(data_dir):
    if file.endswith('.xlsx'):
        file_path = os.path.join(data_dir, file)
        df = read_excel(file_path)
        all_data = pd.concat([all_data, df], ignore_index=True)

# # 划分数据集
# all_data.to_excel(os.path.join(output_dir, 'full.xlsx'), index=False)

# 设置每个类别需要的样本数量
n_samples = 2000

# 对类别字段进行重采样，保证每个取值的数量都为n_samples, 不足n_samples个则全部保留
resampled_df = all_data.groupby('类别').apply(lambda x: x.sample(n=min(len(x), n_samples), replace=True))
resampled_df = resampled_df.reset_index(drop=True)

print(resampled_df['类别'].value_counts())

train, test = train_test_split(resampled_df, test_size=0.2, random_state=42)
print(f"数据量: {train.shape[0]}, {test.shape[0]}")


# validation, test = train_test_split(temp, test_size=0.5, random_state=42)


def make_json_dataset(df, mode, output_dir='./data/full/'):
    # 提取类别和简介字段
    sub_df = df[['类别', '简介']]

    # 转换为字典列表
    data_list = sub_df.to_dict('records')

    # 构建json格式的数据集
    json_data = []
    for i, row in enumerate(data_list):
        json_row = {
            'id': i,
            'input': row['简介'],
            'output': row['类别']
        }
        json_data.append(json_row)

    # 将json数据写入文件
    output_path = output_dir + mode + '.jsonl'
    with open(output_path, 'w') as f:
        for line in json_data:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
    print(f"成功写入到{output_path}")


make_json_dataset(train, "train")
make_json_dataset(test, 'test')

# 输出训练集、验证集和测试集到excel文件
# train.to_excel(os.path.join(output_dir, 'train.xlsx'), index=False)
# validation.to_excel(os.path.join(output_dir, 'val.xlsx'), index=False)
# test.to_excel(os.path.join(output_dir, 'test.xlsx'), index=False)

# 统计类别字段的不同取值的数量
# category_counts = all_data['类别'].value_counts()
# data = [(key, value) for key, value in zip(category_counts.index, category_counts.values)]
# print(data)

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.figure(figsize=(12, 8))
# # colors = (matplotlib.colormaps.get_cmap("Set2")
# plt.pie(category_counts.values, labels=category_counts.index, 
#         autopct='%1.1f%%', startangle=90, labeldistance=1.1)
# plt.title('不同类别小说的数量')
# plt.legend(loc='upper right')
# plt.axis('equal')
# plt.show()
