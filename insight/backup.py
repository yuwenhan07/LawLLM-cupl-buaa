import json
import difflib
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig
import streamlit as st
import re
import os

st.set_page_config(page_title="INSIGHT洞见", page_icon=":robot_face:")
from tqdm import tqdm
from openai import OpenAI

from itertools import islice
os.environ['OPENAI_API_KEY'] = "empty"
client = OpenAI()
client.base_url = "http://127.0.0.1:20000/v1"
model = "baichuan2-13b-chat"
# from your_module import read_jsonl  # Assuming `read_jsonl` is a function you have defined
def chatchat(content):
    completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}]
        )
    response_text = completion.choices[0].message.content
    return response_text

    


st.image('/home/guanhua/chatchat/Langchain-Chatchat/insight/logoo_transparent.png', width=180)
st.title("INSIGHT洞见")
# print(torch.cuda.device_count())
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1，2,3,4'  # 使用第1、2、3块GPU（从0开始计数）

# @st.cache_resource
# def init_model():
#     path = 'baichuan-inc/Baichuan2-13B-Chat'
#     tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16,
#                                                  trust_remote_code=True)
#     model.generation_config = GenerationConfig.from_pretrained(path)
#     return model, tokenizer



def get_description():
    # st.markdown('<div style="background-color:#ffcccb; padding:10px; border-radius:5px;">'
    #             '<h4>请输入案件描述</h4></div>', unsafe_allow_html=True)
    description = st.text_area("请输入案件描述")
    return description


def predict_crime(case_info):
    des = "请你预测以下案件可能性最大的罪名,仅给出罪名即可，\n案件描述如下:\n" + case_info["description"]
    # message_des = [{"role": "user", "content": des}]
    crime = chatchat(des)
    return crime


def match_law(case_info):
    with open('/home/guanhua/chatchat/Langchain-Chatchat/insight/reference/crime-mapping.json') as f:
        law_text = json.load(f)
    # 条文匹配
    closest_match = difflib.get_close_matches(case_info["crime"], law_text.keys(), n=1, cutoff=0.4)
    if closest_match:
        law = law_text[closest_match[0]]
    else:
        law = ""
    return law


def generate_elements(case_info):
    ele = "对于以下法条：\n" + case_info["law"] + "\n请分析其案件构成要件，包括危害行为，行为对象，危害结果，犯罪主体，因果关系以及故意过失"
    # message_law = [{"role": "user", "content": ele}]
    elements = chatchat(ele)
    return elements


def analyze_elements(case_info):
    ana = "对于以下案件:" + case_info["description"] + "请你对照构成要件信息，逐条比对是否符合构成要件要求。构成要件：" + \
          case_info["elements"]
    # message_ana = [{"role": "user", "content": ana}]
    analyze =chatchat(ana)
    return analyze


def analyze_legality(case_info):
    emergency = "紧急避险的成立条件：1.起因条件：存在现实的危险 2.时间条件：危险正在发生 3.主观条件：避险人是否需要避险意思  4.“不得已”条件：避险手段只能是不得已而为之，没有其他合理方法可以选择 5.限度条件：所保护的法益必须大于其损害的法益"
    defence = "正当防卫的成立条件：1.起因条件：面临的侵害具有不法性、客观性和现实性。 2.时间条件：不法侵害正在进行，对于不法侵害是否已经开始或者结束，要立足防卫人在防卫时所处情景，按照社会公众的一般认知，依法作出合乎情理的判断，不能苛求防卫人。3.意思条件：防卫者是否具有防卫的主观意思。 4.对象条件：防卫手段针对不法侵害人本人 5.限度条件：防卫手段具有必要性和相当性"
    elimination = "违法阻却事由一般包括正当防卫、紧急避险、法令行为、正当业务行为、自救行为、被害人承诺、推定承诺、假定承诺、自损行为、危险接受、义务冲突等。"
    legality = emergency + defence + "对于以下案件描述，判断其是否存在以下的违法阻却事由\n，先进行判断，若没有明显现象直接输出无，若有，先判断类型，然后根据后附的案件要素进行分析。\n违法阻却事由：" + elimination + "案件描述" \
               + case_info["description"] + "\n正当防卫构成要件：" + defence + "\n紧急避险构成要件：" + emergency
    # message_legality = [{"role": "user", "content": legality}]
    legality_ana = chatchat(legality)
    return legality_ana


def analyze_responsibility(case_info):
    law_origin1 = '已满十六周岁的人犯罪，应当负刑事责任。已满十四周岁不满十六周岁的人，犯故意杀人、故意伤害致人重伤或者死亡、\
    强奸、抢劫、贩卖毒品、放火、爆炸、投放危险物质罪的，应当负刑事责任。已满十四周岁不满十八周岁的人犯罪，应当从轻或者减轻处罚。因不满十六周岁不予刑事处罚的，责令他的家长或者监护人加以管教；在必要的时候，也可以由政府收容教养。已满七十五周岁的人故意犯罪的，可以从轻或者减轻处罚。'
    law_origin2 = "第十八条 【特殊人员的刑事责任能力】精神病人在不能辨认或者不能控制自己行为的时候造成危害结果，经法定程序鉴定确认的，不负刑事责任，但是应当责令他的家属或者监护人严加看管和医疗;在必要的时候，由政府强制医疗。间歇性的精神病人在精神正常的时候犯罪，应当负刑事责任。尚未完全丧失辨认或者控制自己行为能力的精神病人犯罪的，应当负刑事责任，但是可以从轻或者减轻处罚。醉酒的人犯罪，应当负刑事责任。第十九条 【又聋又哑的人或盲人犯罪的刑事责任】又聋又哑的人或者盲人犯罪，可以从轻、减轻或者免除处罚。"
    responsibility = "根据以下案件描述，对被告人有责性进行分析，具体包括刑事责任年龄、刑事责任能力、违法性认识可能和期待可能性。\n案件信息：" + case_info["description"] \
                     + "\n刑事责任年龄有关条款：" + law_origin1 + "\n刑事责任能力条款：" + law_origin2
    responsibility_ana = chatchat(responsibility)
    return responsibility_ana


def get_penalty_instruction(case_info):
    with open('/home/guanhua/chatchat/Langchain-Chatchat/insight/reference/punishment-mapping.json') as f:
        punish_refer = json.load(f)
    closest_match = difflib.get_close_matches(case_info["crime"], punish_refer.keys(), n=1, cutoff=0.7)
    if closest_match:
        instruction = punish_refer[closest_match[0]]
    else:
        instruction = ""
    return instruction


def calculate_base_penalty(case_info):
    base = "请根据以下参考信息，确定该案件的量刑起点，你并【不需要考虑】与案件事实本身无关的所有加减刑因素，只需要对照\
    法条和指导意见，第一步，根据基本犯罪构成事实在相应的法定刑幅度内确定量刑起点。第二步，根据其他影响犯罪构成的犯罪数额、犯罪次数、犯罪后果等犯罪事实，在量刑起点的基础上增加刑罚量确定基准刑。\n案件信息：" + \
           case_info["description"] + "\n刑法条款:" + case_info["law"] + "\n量刑指导" + case_info["instruction"]
    # message_base = [{"role": "user", "content": base}]
    base_ana = chatchat(base)
    return base_ana


def analyze_extra_penalty_factors(case_info):
    true_content = []
    ages = "请根据以下案件描述,输出罪犯年龄,只要输出一个单一的数字就可以！" + case_info[
        "description"] + "只要给出一个数字即可,请不要自行推断"
    # message_fac1 = [{"role": "user", "content": ages}]
    age = chatchat(ages)
    if int(age) <= 18:
        true_content.append("未成年人犯罪")
    elif int(age) >= 75:
        true_content.append("老年人犯罪")
    fac2 = "请根据以下案件描述,结合“未遂：已经着手实行犯罪,由于犯罪分子意志之外的原因而未得逞的”,给出以下要素的存在结果，如果未明确提及请回答FALSE，请勿自行进行任何推断：\n{罪犯为残疾人：bool，未遂：bool，从犯：bool，自首：bool，积极坦白罪行：bool，退赃退赔：bool，立功：bool，赔偿损失：bool，和解：bool}" + \
           case_info["description"] + "\n注意：明确提及的才为TRUE ，否则均为FALSE。！请不要自行推断！"
    # message_fac2 = [{"role": "user", "content": fac2}]
    ans = chatchat(fac2)
    # print(ans)
    ans = ans.replace("\n", ",")
    ans = ans.replace("{", ",")
    ans = ans.replace("罪犯为残疾人", "残疾人犯罪")
    ans = ans.replace("赔偿损失", "赔偿或谅解")
    ans = ans.replace("积极坦白罪行", "坦白")

    # 记录输出结果bool类型为true的内容
    # true_content = []
    matches = re.findall(r",([^,]*): TRUE", ans)
    for match in matches:
        true_content.append(match.strip())
    matches = re.findall(r",([^,]*): True", ans)
    for match in matches:
        true_content.append(match.strip())
    matches = re.findall(r",([^,]*): true", ans)
    for match in matches:
        true_content.append(match.strip())
    # print(true_content)

    # print("-------------------------------------------")

    fac3 = "请根据以下案件描述,给出以下要素的存在结果，如果未明确提及请回答FALSE，请勿自行进行任何推断：\n{多次犯罪：bool,在本次犯罪行为出现前已有犯罪行为：bool,行为对象为未成年人、老年人、残疾人、孕妇等弱势人员：bool,在重大自然灾害、预防、控制突发传染病疫情等灾害期间故意犯罪：bool}" + \
           case_info["description"] + "\n注意：明确提及的才为TRUE，否则均为FALSE。！请不要自行推断！"
    # message_fac3 = [{"role": "user", "content": fac3}]
    add = chatchat(fac3)
    add = "," + add
    add = add.replace("\n", ",")
    add = add.replace("多次犯罪", "累犯")
    add = add.replace("在本次犯罪行为出现前已有犯罪行为", "前科")
    add = add.replace("行为对象为未成年人、老年人、残疾人、孕妇等弱势人员", "侵害弱势人员权利")
    add = add.replace("在重大自然灾害、预防、控制突发传染病疫情等灾害期间故意犯罪", "自然灾害期间犯罪")
    # print(add)
    # print("-----------------------------------------------")
    # print(true_content)
    true_content_1 = []
    matches = re.findall(r",([^,]*): TRUE", add)
    for match in matches:
        true_content_1.append(match.strip())
    matches = re.findall(r",([^,]*): True", add)
    for match in matches:
        true_content_1.append(match.strip())
    matches = re.findall(r",([^,]*): true", add)
    for match in matches:
        true_content_1.append(match.strip())
    return true_content, true_content_1


def get_sentencing_factor_instruction(case_info):
    with open('/home/guanhua/chatchat/Langchain-Chatchat/insight/reference/factors-mapping.json', encoding='utf-8') as factor_file:
        factor_mapping_data = json.load(factor_file)
    add_sentencing_factors = []
    # 检索生成的内容并输出相应条款
    if case_info["decrease"]:
        for content in case_info["decrease"]:
            if content in factor_mapping_data:
                # print(f"对应的内容为: {factor_mapping_data[content]}")
                add_sentencing_factors.append(f"{content}:")
                add_sentencing_factors.append(f"{factor_mapping_data[content]}")

    if case_info["increase"]:
        for content in case_info["increase"]:
            if content in factor_mapping_data:
                # print(f"对应的内容为: {factor_mapping_data[content]}")
                add_sentencing_factors.append(f"{content}:")
                add_sentencing_factors.append(f"{factor_mapping_data[content]}")
    return add_sentencing_factors


def calculate_final_penalty(case_info):
    pun = "请参考以下加减刑条例、基准刑分析和案件描述，给出详细的【计算过程】和最终的量刑意见\n案件描述：" \
          + case_info["description"] + "\n基准刑分析：" + case_info["base"] + "\n并有如下加减刑因素，加减刑条例：" + str(case_info["sentencing_factors_law"]) \
          + "\n注意，你需要给出详尽的计算和最终结果。"
    # message_pun = [{"role": "user", "content": pun}]
    punishments = chatchat(pun)
    return punishments

animation_html = '''
<div class="loading-animation">
    <div class="head"></div>
    <div class="foot" id="foot1"></div>
    <div class="foot" id="foot2"></div>
</div>

<style>
.loading-animation {
    display: inline-block;
    position: relative;
    width: 50px;
    height: 20px;
    margin-left: 20px; /* Adjust as needed */
}

.head {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #007bff;
    position: absolute;
    top: 0;
    left: 15px;
    animation: bounce 0.5s infinite alternate;
}

.foot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #007bff;
    position: absolute;
    bottom: 0;
}

#foot1 {
    left: 0;
    animation: runFoot 0.5s infinite;
}

#foot2 {
    right: 0;
    animation: runFoot 0.5s infinite 0.25s;
}

@keyframes bounce {
    to {
        transform: translateY(-15px);
    }
}

@keyframes runFoot {
    0%, 100% {
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
}
</style>
'''
# def generate_judgement(case_info):

#     return


def main():
    case_info = {
        "description": "",
        "law": "",
        "crime": "",
        "elements": "",
        "analyze": "",
        "legality": "",
        "responsibility": "",
        "base": "",
        "punishments": "",
        "instruction": "",
        "increase": "",
        "decrease": "",
        "sentencing_factors_law": ""
    }

    # model, tokenizer = init_model()
    description = get_description()
    case_info["description"] = description

    if description:
        # 首先，显示案件描述
        st.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>案件描述</h3>'
                    '<p>{}</p></div>'.format(case_info["description"]), unsafe_allow_html=True)

        # 为后续的各部分分析创建占位符
        placeholder_crime = st.empty()
        placeholder_law_sidebar = st.sidebar.empty()  # 刑法条款将放在侧边栏
        

        # 显示正在分析的文本
        placeholder_crime.markdown(f'''
    <div style="background-color:#ffcccb; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析罪名</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        
        # placeholder_law_sidebar.markdown("正在匹配刑法条款...")
        

        # 执行罪名预测
        case_info["crime"] = predict_crime(case_info)
        # 使用结果替换占位符中的文本
        placeholder_crime.markdown('<div style="background-color:#ffcccb; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                                '<h3>罪名</h3>'
                                '<p>{}</p></div>'.format(case_info["crime"]), unsafe_allow_html=True)

        # 匹配刑法条款
        case_info["law"] = match_law(case_info)
        # 将结果显示在侧边栏，并替换之前的占位符文本
        # placeholder_law_sidebar.title('刑法条款')
        st.sidebar.title('刑法条款')
        st.sidebar.text(case_info["law"])
        # placeholder_law_sidebar.text(case_info["law"])
        placeholder_elements = st.empty()
        placeholder_elements.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析构成要件</h3><br>
    {animation_html}''', unsafe_allow_html=True)
    #     placeholder_crime.markdown(f'''
    # <div style="background-color:#ffcccb; padding:10px; border-radius:5px; margin-bottom: 10px;">
    # <h3 style="font-size: 14px;">正在分析罪名</h3><br>
    # {animation_html}''', unsafe_allow_html=True)
        # 生成构成要件
        case_info["elements"] = generate_elements(case_info)
        # 使用结果替换占位符中的文本
        placeholder_elements.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                                    '<h3>构成要件</h3>'
                                    '<p>{}</p></div>'.format(case_info["elements"]), unsafe_allow_html=True)


        # 假设analyze_elements, analyze_legality, 和 analyze_responsibility函数已经被定义

        # 使用st.columns创建三列
        col1, col2, col3 = st.columns(3)

        # 在每列中创建一个空白占位符
        placeholder1 = col1.empty()
        placeholder2 = col2.empty()
        placeholder3 = col3.empty()

        # 在占位符中显示"正在分析"的消息
        placeholder1.markdown(f'''
    <div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析构成要件该当性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        placeholder2.markdown('''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析违法性</h3><br>
    ''', unsafe_allow_html=True)
        placeholder3.markdown('''
    <div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析有责性</h3><br>
    ''', unsafe_allow_html=True)

        # 执行分析函数
        
        case_info["analyze"] = analyze_elements(case_info)
        # 使用实际内容替换占位符
        placeholder1.markdown('<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>构成要件该当性</h3>'
                            '<p>{}</p></div>'.format(case_info["analyze"]), unsafe_allow_html=True)
        placeholder2.markdown(f'''
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析违法性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["legality"] = analyze_legality(case_info)
        placeholder2.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>违法性分析</h3>'
                            '<p>{}</p></div>'.format(case_info["legality"]), unsafe_allow_html=True)
        placeholder3.markdown(f'''
<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析有责性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["responsibility"] = analyze_responsibility(case_info)
        placeholder3.markdown('<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>有责性分析</h3>'
                            '<p>{}</p></div>'.format(case_info["responsibility"]), unsafe_allow_html=True)


            
        case_info["instruction"] = get_penalty_instruction(case_info)

        st.sidebar.title('量刑指导')
        st.sidebar.text(case_info["instruction"])
        placeholder_base = st.empty()
        placeholder_base.markdown(f'''
    <div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析基准刑</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["base"] = calculate_base_penalty(case_info)
        placeholder_base.markdown('<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>基准刑</h3>'
                    '<p>{}</p></div>'.format(case_info["base"]), unsafe_allow_html=True)
        colup, coldown = st.columns(2)

        # 在每列中创建一个空白占位符
        placeholderup = colup.empty()
        placeholderdown = coldown.empty()
        # placeholder3 = col3.empty()
        placeholderup.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析减刑因素</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        placeholderdown.markdown(f'''
    <div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析增刑因素</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        
        case_info["decrease"], case_info["increase"] = analyze_extra_penalty_factors(case_info)
        placeholderup.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>减刑因素</h3>'
                    '<p>{}</p></div>'.format(case_info["decrease"]), unsafe_allow_html=True)
        placeholderdown.markdown('<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>增刑因素</h3>'
                    '<p>{}</p></div>'.format(case_info["increase"]), unsafe_allow_html=True)

        case_info["sentencing_factors_law"] = get_sentencing_factor_instruction(case_info)
        formatted_factors = "\n".join(case_info["sentencing_factors_law"])

    
        st.sidebar.title('加减刑指导意见')
        st.sidebar.text(formatted_factors)
        placeholder_final=st.empty()
        placeholder_final.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析最终刑罚</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["punishments"] = calculate_final_penalty(case_info)
        placeholder_final.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>刑罚</h3>'
                    '<p>{}</p></div>'.format(case_info["punishments"]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()