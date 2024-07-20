import json
import random
from datetime import datetime, timedelta

# 生成随机日期
def random_date(start, end):
    return start + timedelta(days=random.randint(0, int((end - start).days)))

# 生成随机的条目
def generate_random_entries(num_entries):
    entries = []
    suspects = ["李某", "王某", "张某", "刘某某", "杨某", "王某某", "张某某", "李某某", "陈某", "赵某", "黄某", "周某"]
    victims = ["王某某", "刘某某", "周某某", "陈某某", "孙某某", "赵某某", "李某某", "张某某", "杨某某", "黄某某"]
    locations = ["东风小区", "华光小区", "富华小区", "新村小区", "康乐小区", "翠园小区", "南湖小区", "长兴小区", "福安小区", "蓝天小区", "春风小区", "红星小区"]
    items = ["苹果手机一部", "现金500元", "电动车一辆", "笔记本电脑一台", "金项链一条", "摩托车一辆", "自行车一辆", "平板电脑一台", "手表一只", "珠宝一件", "照相机一台", "钱包一个"]

    start_date = datetime(2017, 1, 1)
    end_date = datetime(2023, 7, 1)

    for i in range(num_entries):
        entry_id = f"a{i+1}"
        date_time = random_date(start_date, end_date).strftime("%Y年%m月%d日%H点%M分")
        suspect = random.choice(suspects)
        victim = random.choice(victims)
        location = random.choice(locations)
        item = random.choice(items)
        value = f"{random.randint(500, 5000)}余元"

        if i % 2 == 0:
            context = f"被害人报案总价值{value}。"
            entities = [{"label": "物品价值", "text": value}]
        else:
            context = f"{date_time}，被告人{suspect}在{location}砸碎被害人{victim}的车窗玻璃，盗窃{item}。"
            entities = [
                {"label": "犯罪嫌疑人", "text": suspect},
                {"label": "受害人", "text": victim},
                {"label": "被盗物品", "text": item},
                {"label": "时间", "text": date_time},
                {"label": "地点", "text": location}
            ]

        entry = {"id": entry_id, "context": context, "entities": entities}
        entries.append(entry)

    return entries

# 生成100个条目
data = generate_random_entries(100)

# 保存为JSON文件
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)