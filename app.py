import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="NER+关系抽取+知识图谱可视化",
    page_icon="🕸️",
    layout="wide"
)

# ---------------------- 内置实体词典（模拟NER） ----------------------
ENTITY_DICT = {
    "PERSON": ["Steve Jobs", "Elon Musk", "Bill Gates", "Tim Cook"],
    "ORG": ["Apple", "Google", "Microsoft", "Tesla", "Apple Inc."],
    "GPE": ["Cupertino", "California", "USA", "United States"],
    "PRODUCT": ["iPhone", "MacBook", "Tesla Model S"]
}

def get_ner_entities(text):
    entities = []
    text_lower = text.lower()
    for label, names in ENTITY_DICT.items():
        for name in names:
            name_lower = name.lower()
            start_idx = text_lower.find(name_lower)
            while start_idx != -1:
                entities.append({
                    "text": name,
                    "start": start_idx,
                    "end": start_idx + len(name),
                    "label": label
                })
                start_idx = text_lower.find(name_lower, start_idx + len(name))
    # 按出现位置倒序排序，避免高亮时的偏移问题
    return sorted(entities, key=lambda x: x["start"], reverse=True)

def get_bio_tags(text, entities):
    bio_tags = []
    # 初始化所有token为O
    tokens = text.split()
    for token in tokens:
        bio_tags.append((token, "O"))
    
    # 标记实体的B/I标签
    for ent in entities:
        ent_tokens = ent["text"].split()
        for i, token in enumerate(ent_tokens):
            for j, (t, _) in enumerate(bio_tags):
                if t == token:
                    if i == 0:
                        bio_tags[j] = (t, f"B-{ent['label']}")
                    else:
                        bio_tags[j] = (t, f"I-{ent['label']}")
    return bio_tags

def highlight_entities(text, entities):
    for ent in entities:
        label = ent["label"]
        color_map = {
            "PERSON": "#FFB3BA",
            "ORG": "#BAFFC9",
            "GPE": "#BAE1FF",
            "PRODUCT": "#FFD9B3"
        }
        color = color_map.get(label, "#E0E0E0")
        html = f'<mark style="background-color: {color}; padding: 2px; border-radius: 3px;">{ent["text"]} <sub style="font-size: 0.7em; color: black;">{label}</sub></mark>'
        text = text[:ent["start"]] + html + text[ent["end"]:]
    return text

# ---------------------- 简单关系抽取 ----------------------
def extract_relations(text, entities):
    relations = []
    entity_texts = [ent["text"] for ent in entities]
    if len(entity_texts) < 2:
        return relations
    
    relation_patterns = {
        "founder": ["founded", "founder of", "created by"],
        "located in": ["in", "located in", "based in"],
        "works at": ["works at", "employed by", "at"]
    }
    
    for i in range(len(entity_texts)):
        for j in range(len(entity_texts)):
            if i == j:
                continue
            subj = entity_texts[i]
            obj = entity_texts[j]
            # 简单判断实体顺序
            subj_idx = text.find(subj)
            obj_idx = text.find(obj)
            if subj_idx < obj_idx:
                middle_text = text[subj_idx + len(subj):obj_idx].lower()
                for rel, keywords in relation_patterns.items():
                    for kw in keywords:
                        if kw in middle_text:
                            relations.append({
                                "subject": subj,
                                "predicate": rel,
                                "object": obj
                            })
    return relations

# ---------------------- 知识图谱可视化 ----------------------
def build_knowledge_graph(relations, entities):
    net = Network(notebook=False, height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    entity_color_map = {
        "PERSON": "#FFB3BA",
        "ORG": "#BAFFC9",
        "GPE": "#BAE1FF",
        "PRODUCT": "#FFD9B3"
    }
    node_labels = set()
    for ent in entities:
        if ent["text"] not in node_labels:
            net.add_node(
                ent["text"], 
                label=ent["text"], 
                color=entity_color_map.get(ent["label"], "#E0E0E0"),
                title=ent["label"]
            )
            node_labels.add(ent["text"])
    
    for rel in relations:
        net.add_edge(
            rel["subject"], 
            rel["object"], 
            label=rel["predicate"],
            title=rel["predicate"],
            color="#FFFFFF"
        )
    
    net.save_graph("kb_graph.html")
    return open("kb_graph.html", "r", encoding="utf-8").read()

# ---------------------- 页面内容 ----------------------
st.title("🕸️ NER+关系抽取+知识图谱可视化系统")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "模块1：NER与BIO标注",
    "模块2：实体关系抽取",
    "模块3：知识图谱可视化"
])

# ---------------------- 模块1：NER与BIO标注 ----------------------
with tab1:
    st.header("🔍 命名实体识别（NER）与BIO标注")
    st.markdown("输入英文文本，高亮显示实体并查看BIO标注结果")
    
    text_input = st.text_area(
        "请输入英文句子：",
        value="Steve Jobs founded Apple Inc. in Cupertino.",
        height=150,
        key="ner_text"
    )
    show_bio = st.checkbox("开启查看底层标注模式（BIO）", value=False, key="bio_check")
    
    if st.button("开始识别", key="ner_btn"):
        entities = get_ner_entities(text_input)
        st.subheader("识别结果（高亮实体）")
        st.markdown(highlight_entities(text_input, entities), unsafe_allow_html=True)
        
        if show_bio:
            st.subheader("BIO标注结果")
            bio_tags = get_bio_tags(text_input, entities)
            bio_text = " ".join([f"{token} ({tag})" for token, tag in bio_tags])
            st.code(bio_text)
        
        st.subheader("实体列表")
        st.dataframe(pd.DataFrame(entities))

# ---------------------- 模块2：实体关系抽取 ----------------------
with tab2:
    st.header("🔗 实体关系抽取（Relation Extraction）")
    st.markdown("在NER结果基础上，抽取实体间的关系")
    
    text_input2 = st.text_area(
        "请输入英文句子：",
        value="Steve Jobs founded Apple Inc. in Cupertino.",
        height=150,
        key="rel_text"
    )
    
    if st.button("抽取关系", key="rel_btn"):
        entities = get_ner_entities(text_input2)
        relations = extract_relations(text_input2, entities)
        
        st.subheader("实体列表")
        st.dataframe(pd.DataFrame(entities))
        
        st.subheader("关系抽取结果")
        if relations:
            st.dataframe(pd.DataFrame(relations))
        else:
            st.info("未抽取到实体间关系，请尝试包含明确关系词的句子（如 'founded by', 'located in'）")

# ---------------------- 模块3：知识图谱可视化 ----------------------
with tab3:
    st.header("🕸️ 知识图谱交互式可视化")
    st.markdown("将实体与关系转换为可交互的知识图谱，支持拖拽和缩放")
    
    text_input3 = st.text_area(
        "请输入英文句子：",
        value="Steve Jobs founded Apple Inc. in Cupertino. Apple is a technology company.",
        height=150,
        key="kb_text"
    )
    
    if st.button("生成知识图谱", key="kb_btn"):
        entities = get_ner_entities(text_input3)
        relations = extract_relations(text_input3, entities)
        
        if not relations:
            st.warning("未抽取到实体间关系，图谱将仅显示实体节点")
        
        html_content = build_knowledge_graph(relations, entities)
        components.html(html_content, height=600, scrolling=True)

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.markdown("© 2025 NLP 课程 Week X 实验 | NER+关系抽取+知识图谱可视化系统")
