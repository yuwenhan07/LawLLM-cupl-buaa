# 基于RAG与COT的可解释性增强的智能判决预测

## 项目介绍
“INSIGHT洞见”是一个智能司法决策框架，结合Retrieval-Augmented Generation (RAG) 和 Chain-of-Thought (COT) 技术，显著增强法律判决预测的可解释性。项目旨在解决当前法律人工智能模型在生成结果的可解释性、逻辑性和可靠性方面的不足，推动智能司法在法律行业的应用。

## 项目背景
### 时代背景
- **政策支持**: “人工智能+”被写入政府工作报告，最高法强调加强人工智能应用的顶层设计和智慧法院大脑建设。
- **需求号召**: 法律行业对智能司法的需求日益增加。
- **学术热点**: “AI for Law”成为交叉研究的热门方向。

### 痛点分析
- **可解释性差**: 现有的法律AI模型大多为黑盒模型，缺乏可解释性。
- **推理能力不足**: 小模型在复杂案件中的推理能力有限。
- **数据质量差**: 缺少高质量的训练数据集，影响模型的准确性。
- **结果不确定性高**: 大模型生成的内容具有不确定性，容易出现幻觉。

## 项目核心框架
### 主要功能
- **刑事判决预测**: 基于案件描述预测罪名，并提供详细的法律条款匹配。
- **法律智能问答**: 回答法律相关问题，并提供详细的解释。
- **类案高级查找**: 检索相似案件，辅助法律分析。
- **法律文书撰写**: 自动生成法律文书，提高工作效率。

### 技术创新
1. **鉴定式分析自动判决预测**: 提供更高准确度和专业性的因果推理链条。
2. **多视角类案检索**: 利用查询重写、多视角矩阵构建、结果重排技术，实现全面优化，提升召回率与准确率。

### 可解释性增强
- **罪名-要件标签集**: 通过标签集匹配罪名与法条，提供清晰的解释。
- **三阶层断案模型**: 包括构成要件、法条匹配和案外情节，确保逻辑性和可解释性。
- **量刑计算模型**: 通过基准刑分析、增减刑因素分析，提供详细的量刑计算过程。

## 项目实现
### 框架搭建与技术实现
- **数据预处理**: 利用指令微调、全量微调、LoRA微调、知识强化等方法进行模型训练。
- **专业知识库检索**: 嵌入法学机理抽象编码技术，构建专业知识库，提高模型的专业性和准确性。
- **UI设计与使用**: 通过Streamlit进行UI设计，提供用户友好的界面。

### 流程与代码示例
1. **案件描述输入**: 用户输入案件描述，系统自动分析。
2. **罪名预测**: 系统基于描述预测罪名，并匹配相关法律条款。
3. **构成要件分析**: 提供详细的构成要件分析，包括危害行为、行为对象等。
4. **违法性与有责性分析**: 对照构成要件，分析案件的违法性和有责性。
5. **量刑指导**: 提供详细的量刑指导和计算过程。

### 代码示例
以下是项目核心代码示例：

```python
import json
import difflib
import torch
import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置设备为CUDA
device = "cuda"

# 加载本地的GLM-4-9b-chat模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("path_to_model", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("path_to_model", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

st.set_page_config(page_title="INSIGHT洞见", page_icon=":robot_face:", layout="wide")

def chatchat(content):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 1000, "do_sample": True, "top_k": 60}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

st.title("INSIGHT洞见")
st.image('logoo_transparent.png', width=180)

def get_description():
    description = st.text_area("请输入案件描述")
    return description

def predict_crime(case_info):
    des = "请你预测以下案件可能性最大的罪名,仅给出罪名即可，\n案件描述如下:\n" + case_info["description"]
    crime = chatchat(des)
    return crime

def match_law(case_info):
    with open('reference/crime-mapping.json', encoding='utf-8') as f:
        law_text = json.load(f)
    closest_match = difflib.get_close_matches(case_info["crime"], law_text.keys(), n=1, cutoff=0.4)
    law = law_text[closest_match[0]] if closest_match else ""
    return law

# 其他功能函数...

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

    description = get_description()
    case_info["description"] = description

    if description:
        st.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>案件描述</h3>'
                    '<p>{}</p></div>'.format(case_info["description"]), unsafe_allow_html=True)

        placeholder_crime = st.empty()
        placeholder_law_sidebar = st.sidebar.empty()
        
        placeholder_crime.markdown(f'''
    <div style="background-color:#f6d9d9; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析罪名</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        
        case_info["crime"] = predict_crime(case_info)
        placeholder_crime.markdown('<div style="background-color:#f6d9d9; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                                '<h3>罪名</h3>'
                                '<p>{}</p></div>'.format(case_info["crime"]), unsafe_allow_html=True)

        case_info["law"] = match_law(case_info)
        st.sidebar.title('刑法条款')
        st.sidebar.markdown('<div style="word-wrap: break-word;">'
            '<p>{}</p></div>'.format(case_info["law"]), unsafe_allow_html=True)

        placeholder_elements = st.empty()
        placeholder_elements.markdown(f'''
    <div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析构成要件</h3><br>
    {animation_html}''', unsafe_allow_html=True)

        case_info["elements"] = generate_elements(case_info)
        placeholder_elements.markdown('<div style="background-color:#add8e6; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                                    '<h3>构成要件</h3>'
                                    '<p>{}</p></div>'.format(case_info["elements"]), unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        placeholder1 = col1.empty()
        placeholder2 = col2.empty()
        placeholder3 = col3.empty()
        
        placeholder1.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析构成要件该当性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        placeholder2.markdown('''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析违法性</h3><br>
    ''', unsafe_allow_html=True)
        placeholder3.markdown('''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析有责性</h3><br>
    ''', unsafe_allow_html=True)

        case_info["analyze"] = analyze_elements(case_info)
        shortened_analyze = case_info["analyze"][:125] + "..." + case_info["analyze"][-125:]
        placeholder1.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>构成要件该当性</h3>'
                            '<p>{}</p></div>'.format(shortened_analyze), unsafe_allow_html=True)

        placeholder2.markdown(f'''
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析违法性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["legality"] = analyze_legality(case_info)
        shortened_legality = case_info["legality"][:150] + "..." + case_info["legality"][-150:]
        placeholder2.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>违法性分析</h3>'
                            '<p>{}</p></div>'.format(shortened_legality), unsafe_allow_html=True)

        placeholder3.markdown(f'''
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析有责性</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["responsibility"] = analyze_responsibility(case_info)
        shortened_responsibility = case_info["responsibility"][:160] + "..." + case_info["responsibility"][-160:]
        placeholder3.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                            '<h3>有责性分析</h3>'
                            '<p>{}</p></div>'.format(shortened_responsibility), unsafe_allow_html=True)

        case_info["instruction"] = get_penalty_instruction(case_info)
        st.sidebar.title('量刑指导')
        st.sidebar.markdown('<div style="word-wrap: break-word;">'
            '<p>{}</p></div>'.format(case_info["instruction"]), unsafe_allow_html=True)

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
        placeholderup = colup.empty()
        placeholderdown = coldown.empty()

        placeholderup.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析减刑因素</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        placeholderdown.markdown(f'''
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析增刑因素</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        
        case_info["decrease"], case_info["increase"] = analyze_extra_penalty_factors(case_info)
        placeholderup.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>减刑因素</h3>'
                    '<p>{}</p></div>'.format(case_info["decrease"]), unsafe_allow_html=True)
        placeholderdown.markdown('<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>增刑因素</h3>'
                    '<p>{}</p></div>'.format(case_info["increase"]), unsafe_allow_html=True)

        case_info["sentencing_factors_law"] = get_sentencing_factor_instruction(case_info)
        formatted_factors = "\n".join(case_info["sentencing_factors_law"])
        st.sidebar.title('加减刑指导意见')
        st.sidebar.markdown('<div style="word-wrap: break-word;">'
            '<p>{}</p></div>'.format(formatted_factors), unsafe_allow_html=True)

        placeholder_final=st.empty()
        placeholder_final.markdown(f'''
    <div style="background-color:#f6d9d9; padding:10px; border-radius:5px; margin-bottom: 10px;">
    <h3 style="font-size: 14px;">正在分析最终刑罚</h3><br>
    {animation_html}''', unsafe_allow_html=True)
        case_info["punishments"] = calculate_final_penalty(case_info)
        placeholder_final.markdown('<div style="background-color:#f6d9d9; padding:10px; border-radius:5px; margin-bottom: 10px;">'
                    '<h3>刑罚</h3>'
                    '<p>{}</p></div>'.format(case_info["punishments"]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
```

## 项目成果
- **20+法律大模型调研**
- **10+法律模型适配应用**
- **5项功能本地化部署测试**
- **项目网站上线**
- **专业技术论文投稿**
