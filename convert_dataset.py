from datasets import load_dataset
import json
import string
import nltk
# 设置代理并加载 SentiHood 数据集
import os
os.environ["http_proxy"] = "http://127.0.0.1:7860"
os.environ["https_proxy"] = "http://127.0.0.1:7860"
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer



# 加载数据集
dataset = load_dataset("bhavnicksm/sentihood")

# 查看数据集结构
data = dataset['train']  # 查看前5条训练数据

# 初始化词干化工具
lemmatizer = WordNetLemmatizer()

# 转换为训练代码需要的格式
processed_data = []

for i, sentence in enumerate(data["text"]):
    # 分词并进行词形还原和预处理
    words = [
        lemmatizer.lemmatize(word.lower().strip(string.punctuation))
        for word in sentence.split()
    ]
    aspects = []
    opinions = []

    # 遍历当前句子的所有意见
    for opinion in data["opinions"][i]:
        # 获取方面词
        aspect_term = [
            lemmatizer.lemmatize(term.lower().strip(string.punctuation))
            for term in opinion["aspect"].split()
        ]
        sentiment = opinion["sentiment"]

        # 尝试匹配方面词
        try:
            aspect_start = words.index(aspect_term[0])
            aspect_end = aspect_start + len(aspect_term)
            aspects.append({
                "from": aspect_start,
                "to": aspect_end,
                "term": aspect_term
            })
            opinions.append({
                "from": aspect_start,
                "to": aspect_end,
                "term": aspect_term,
                "sentiment": sentiment
            })
        except ValueError:
            print(f"Aspect term '{aspect_term}' not found in sentence '{sentence}'")
            continue

    # 构造最终的数据结构
    processed_data.append({
        "raw_words": sentence,
        "words": words,
        "task": "AOE",
        "aspects": aspects,
        "opinions": opinions
    })

# 保存为 JSON 文件
output_file = "processed_test_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4)

print(f"数据处理完成，已保存至 {output_file}")