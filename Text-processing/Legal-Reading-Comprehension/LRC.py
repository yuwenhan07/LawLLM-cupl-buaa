import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="../../finetune/output/LRC/checkpoint-500")

test_texts = {
    'instruction': "请你根据下面提供的'法律文本材料'内容，回答相应的'问题'，以完成片段抽取式的阅读理解任务。\n具体来说，你要正确回答'问题'，并且答案限定是'法律文本材料'的一个子句（或片段）。请你以'''答案：A'''的格式给出回答，其中A表示'法律文本材料'中正确的子句（或片段）。",
    'input': """法律文本材料'：本院经审理认定事实如下:原告母亲于2017年1月为原告报名参加了被告华3冰场开班的滑冰课程,课程费3000元(20课时,每次30分钟),教授方x12一对一单独教学,授课期间家长不得进入溜冰场2017年2月6日,原告在被告华3冰场接受滑冰培训,下课后原告在冰场继续滑冰,在向出口处滑冰时不慎将额头撞伤原告受伤后前往沈阳军区总医院急诊治疗,急诊诊断为&quot;额头软组织挫伤、面部软组织挫伤&quot;,建议&quot;1、......;2、伤后抗感染治疗5天;3、术后隔一天首次换药,其后每2-3天换药,7天拆线;4、拆线后抗瘢痕治疗1-3月;5、建议休息贰周;6、病情变化随诊&quot;原告因此次事件花费医疗费2257.34元原告随后到沈阳和平李淼医疗美容诊所进行瘢痕治疗,在沈阳和平李淼医疗美容诊所一次性缴纳20000元治疗及药物费用,另行购买硅凝胶花费650元另查明,原告姥姥徐国华系辽宁众旺诚联合会计师事务所(普通合伙)财务主管,月平均工资6067元,因2017年2月7日至2017年4月15日一直请假未上班,被停发工资12000元再查明,因原告需要治疗,原告母亲张1与王梓屹签订租车协议一份,约定由乙方王梓屹为原告提供车辆租赁接送服务,在2017年2月7日至2017年7月16日期间,发生租车费6000元又查明,被告华3冰场在冰场周围设置意外免责条款、安全提示及入场贴士,意外免责条款中载明&quot;在未成年人进场滑冰前,家长应小心衡量滑冰的风险未成年人在进入冰场前以及在冰面滑冰时应有成人陪同12周岁以下未成年人进场滑冰时,家长或合法监护人可在票房购买陪同卡,每个未成年人只能有一人陪同&quot;上述事实,有原告向法庭提供的急诊病历、医疗费票据、治疗情况说明,误工证明、劳动合同、工资明细、注册会计师证,车辆租赁协议、租车款收条、司机身份证复印件,冰纷万象学员卡、万象城会员卡,原告受伤照片,户口本复印件,被告向法庭提供的报名须知及条款,意外免责条款、入场贴士、安全提示,学员课程明细表及原、被告当庭陈述笔录在卷佐证,已经开庭质证,本院予以确认\n'问题'：租车费是多少？\n"""
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)
