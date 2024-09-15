import re

# 读取数据文件并解析内容
def read_data_from_txt(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.readlines()

        for line in content:
            if '####' in line:
                # 拆分句子和四元组部分
                sentence, quadruples_str = line.split('####')
                sentence = sentence.strip()
                # 将四元组从字符串转换为列表
                quadruples = eval(quadruples_str.strip())
                data.append({"sentence": sentence, "quadruples": quadruples})

    return data


# 情感标签映射
polarity_map = {'0': 'negative', '1': 'neutral', '2': 'positive'}


# 处理四元组并保存到文件
def process_quadruples_and_save(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            sentence = entry["sentence"]
            f.write(f"{sentence}####")  # 写入句子

            processed_quadruples = []  # 用于存储处理后的四元组

            for quadruple in entry["quadruples"]:

                # 1. 处理第一个元素
                first_element = quadruple[0].split('-')[0]

                # 2. 处理第二和第三个元素，提取句子中的索引部分的内容
                second_indices = tuple(map(int, quadruple[1].split(',')))
                second_element = sentence[second_indices[0]:second_indices[1]]

                third_indices = tuple(map(int, quadruple[2].split(',')))
                third_element = sentence[third_indices[0]:third_indices[1]]

                # 3. 处理第四个元素，将其从数字转换为情感标签
                polarity = polarity_map[quadruple[3]]

                # 4. 交换第三个和第四个元素的位置
                processed_quadruple = [second_element, first_element, polarity, third_element]

                processed_quadruple2 = []
                for ele in processed_quadruple:
                    if ele == "":
                        processed_quadruple2.append("NULL")
                    else:
                        processed_quadruple2.append(ele)
                processed_quadruples.append(processed_quadruple2)

            # 写入处理后的四元组，格式为 [四元组1, 四元组2, ...]
            f.write(f"{processed_quadruples}\n")

    print(f"Processed data saved to {output_file}")


# 批量处理多个文件
def process_multiple_files(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        data = read_data_from_txt(input_file)
        process_quadruples_and_save(data, output_file)


# 指定输入和输出文件路径
input_files = ['dev.txt', 'train.txt', 'test.txt']
output_files = ['zh/dev.txt', 'zh/train.txt', 'zh/test.txt']

# 处理并保存多个文件
process_multiple_files(input_files, output_files)
