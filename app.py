import base64
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

import altair
import pandas as pd
import streamlit as st
from pydantic import BaseModel


class ModelsInfo(BaseModel):
    path: str = "models_info.csv"
    info: Dict[str, dict] = {}

    def load(self):
        if not self.info:
            df = pd.read_csv(self.path)
            for record in df.to_dict(orient="records"):
                self.info[record["Model"]] = record

    def get_organization(self, model_name: str) -> str:
        self.load()
        return self.info[model_name]["Organization"]

    def get_website(self, model_name: str) -> str:
        self.load()
        return self.info[model_name]["Website"]

    def get_icon(self, model_name: str) -> str:
        # Streamlit requires the data url eg "data:image/png;base64,iVBO..."
        self.load()
        path = self.info[model_name]["Icon"]
        assert Path(path).exists()

        with open(path, "rb") as f:
            image_data = f.read()
            encoded_data = base64.b64encode(image_data).decode("utf-8")
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".svg": "image/svg+xml",
            }[Path(path).suffix]

            return f"data:{mime_type};base64,{encoded_data}"

    def get_opensource_icon(self, model_name: str) -> str:
        # Streamlit requires the data url eg "data:image/png;base64,iVBO..."
        self.load()
        path = self.info[model_name]["OpenSource"]
        if not Path(str(path)).exists():
            return ""

        with open(path, "rb") as f:
            image_data = f.read()
            encoded_data = base64.b64encode(image_data).decode("utf-8")
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".svg": "image/svg+xml",
            }[Path(path).suffix]

            return f"data:{mime_type};base64,{encoded_data}"

    def get_model_names(self) -> List[str]:
        self.load()
        return sorted(self.info.keys())


def apply_chat_style():
    # Arrange the user and assistant chats in a left-right style (instead of all left)
    # https://discuss.streamlit.io/t/how-to-display-user-and-assistant-chat-on-opposite-sides-when-streaming-like-a-conversation/60336/4
    content = """
    <style>
        .st-emotion-cache-4oy321 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """
    st.markdown(content, unsafe_allow_html=True)


def load_debates_and_judgements(
    debate_file: str, num_per_domain: int = 2
) -> List[tuple]:
    with open(debate_file) as f:
        records = [json.loads(line) for line in f]
        debates = {str(raw["gamekey"][0]): raw for raw in records}
    with open(debate_file.replace("debate_history", "judge_results")) as f:
        records = [json.loads(line) for line in f]
        judgements = {str(raw["gamekey"][0]): raw for raw in records}
    assert debates.keys() == judgements.keys()

    counts = {}
    outputs = []
    for key in sorted(debates.keys()):
        domain = debates[key]["question"]["domain"]
        counts.setdefault(domain, 0)
        if counts[domain] < num_per_domain:
            counts[domain] += 1
            outputs.append((debates[key], judgements[key]))

    return outputs


def find_models(paths: List[Path], info: ModelsInfo) -> List[str]:
    lst = []
    for p in paths:
        for name in info.get_model_names():
            if name.replace("/", "_") in str(p):
                lst.append(name)
    return sorted(set(lst))


def find_opponents(model: str, paths: List[Path], info: ModelsInfo) -> List[str]:
    lst = []
    for p in paths:
        if model.replace("/", "_") in str(p):
            for other in info.get_model_names():
                if other != model and other.replace("/", "_") in str(p):
                    lst.append(other)

    assert lst, breakpoint()
    return sorted(set(lst))


def find_debate_file(model_a: str, model_b: str, paths: List[Path]) -> Path:
    for p in paths:
        if model_a.replace("/", "_") in str(p):
            if model_b.replace("/", "_") in str(p):
                return p


def show_win_rate(path: str):
    judgements = []
    data = load_debates_and_judgements(path, num_per_domain=99)
    for debate, judge in data:
        candidates = debate["candidates"]
        mapping = dict(A=candidates[0], B=candidates[1], tie="Tie", error="Error")
        judgements.append(mapping[judge["final_winner"][-1]])
    st.write(Counter(judgements))


def show_debates(folder: str):
    info = ModelsInfo()
    paths = sorted(Path(folder).glob("round*/*debate_history.jsonl"))
    columns = st.columns(2)
    model_a = columns[0].selectbox("Model", find_models(paths, info))
    model_b = columns[1].selectbox("Opponent", find_opponents(model_a, paths, info))
    debate_file = find_debate_file(model_a, model_b, paths)
    data = load_debates_and_judgements(str(debate_file))
    # show_win_rate(str(debate_file))

    debate, judgements = st.selectbox(
        "Debate Question",
        key=model_a + "_" + model_b,  # This forces the widget and below to update
        options=data,
        format_func=lambda s: f"{s[0]['question']['domain'].capitalize()}: {s[0]['question']['question']}",
    )
    candidates = debate["candidates"]
    with st.chat_message("assistant", avatar="user"):
        st.write(debate["question"]["question"])

    for i, debate_round in enumerate(debate["rounds"]):
        with st.expander(f"Round {i + 1}"):
            for j, (key, turn) in enumerate(debate_round):
                model = dict(a=candidates[0], b=candidates[1])[key]
                if j % 2 == 0:
                    role = "user"
                    col = st.columns([0.9, 0.1])[0]
                else:
                    role = "user"
                    col = st.columns([0.1, 0.9])[1]

                with col.chat_message(role, avatar=info.get_icon(model)):
                    st.link_button(model, info.get_website(model))
                    st.write(turn["original"])

    mapping = dict(A=candidates[0], B=candidates[1], tie="Tie", error="Error")
    with st.expander(f"Overall Winner: {mapping[judgements['final_winner'][-1]]}"):
        judge = st.selectbox("Judge", options=judgements["judges"])
        columns = st.columns(2)
        for j in [0, 1]:
            columns[j].link_button(
                f"Assistant {'AB'[j]}: {candidates[j]}",
                info.get_website(candidates[j]),
                use_container_width=True,
            )

        for k, text in enumerate(judgements[judge]["judgement"]):
            with st.chat_message("user", avatar=info.get_icon(judge)):
                st.write(text)
            winner = mapping[judgements[judge]["winner"][k]]
            if winner not in info.get_model_names():
                continue
            st.link_button(winner, info.get_website(winner), use_container_width=True)


def get_latest_elo_file(folder: str) -> str:
    num = 0
    path = ""
    for p in sorted(Path(folder).iterdir()):
        pattern = r"round(\d+)"
        match = re.search(pattern, p.name)
        if match and int(match.group(1)) > num:
            num = int(match.group(1))
            path = str(p)

    assert path
    return str(Path(path, "elo_history.csv"))


def format_model_info(raw: dict) -> str:
    icon = f'<img class="rounded-icon" src="{raw["Icon"]}"/>'
    model = f'<a href="{raw["Website"]}">{raw["Model"]}</a>'

    open_source = f'<a href="https://opensource.org"> <img class="open-source-icon" src="{raw["OpenSource"]}" title="Open Source"/> </a>'
    if raw["OpenSource"] == "":
        open_source = ""

    return f'{icon} {raw["Organization"]}: {model} {open_source}'


def show_html_table(data: pd.DataFrame):
    data = data.copy(deep=True)

    is_chinese_table = st.sidebar.toggle("Show Chinese Table")
    if is_chinese_table:
        mapping = {
            "Zhipu AI": "智谱 AI",
            "Alibaba Cloud": "阿里云",
            "SenseTime": "商汤",
            "Baidu": "百度",
            "01.AI": "零一万物",
            "Deepseek AI": "深度求索",
        }
        data["Organization"] = data["Organization"].apply(lambda x: mapping.get(x, x))

    data["Model"] = data.apply(format_model_info, axis=1)
    data["Score"] = data["ELO Score"].apply(round)
    data["Rank"] = data["Ranking"]

    raw = data.to_html(
        columns=["Rank", "Model", "Score"],
        justify="left",
        index=False,
        escape=False,
        render_links=True,
    )

    height_setting = """
    height: 400px;
    overflow-y: auto;
    border-radius: 15px; /* Add this line to make the container a rounded rectangle */
    border: 1px solid #dddddd; /* Optional: Add a border to the container */
    """

    if st.sidebar.toggle("Show Full Results Table"):
        height_setting = ""

    style = f"""
    <style>
    a {{
        text-decoration: none; /* Remove underline from hyperlinks */
    }}
    .rounded-icon {{
        border-radius: 8px; /* Adjust this value to change the roundness of the corners */
        width: 28px;
        margin-right: 10px;
    }}
    .open-source-icon {{
        width: 20px;
        margin-left: 10px;
    }}
    table {{
        width: 100%;
        font-size: 14px; /* Set the desired font size for the table */
        border: collapse;
    }}
    th {{
        background-color: #f0f2f6;
        font-weight: normal; /* Add this line to avoid bolding */
    }}
    tr:nth-child(even) {{
        background-color: #f8f9fb;
    }}
    .table-container {{
        {height_setting}
        margin-bottom: 30px; /* Add bottom margin to the container */
    }}
    </style>
    """

    content = f"""
    <div class="table-container">
        {raw}
    </div>
    """

    if is_chinese_table:
        content = content.replace("<th>Rank</th>", "<th>排名</th>")
        content = content.replace("<th>Model</th>", "<th>模型</th>")
        content = content.replace("<th>Score</th>", "<th>评分</th>")

    st.write(style, unsafe_allow_html=True)
    st.write(content, unsafe_allow_html=True)


def show_results(folder: str):
    path = get_latest_elo_file(folder)
    data = pd.read_csv(path)
    data = data.transpose()
    data = data.iloc[1:]
    data = data.reset_index()
    data = data[["index", data.columns[-1]]]
    data.columns = ["Model", "ELO Score"]

    info = ModelsInfo()
    data = data[data["Model"].isin(info.get_model_names())]
    data[data.columns[-1]] = data[data.columns[-1]].round()
    data = data.sort_values(by=data.columns[-1], ascending=False)
    data.insert(0, "Ranking", [i + 1 for i in range(data.shape[0])])
    data.insert(1, "Organization", data["Model"].map(info.get_organization))
    data.insert(3, "Website", data["Model"].map(info.get_website))
    data.insert(1, "Icon", data["Model"].map(info.get_icon))
    data.insert(0, "OpenSource", data["Model"].map(info.get_opensource_icon))
    data["Model"] = data["Model"].apply(lambda x: x.split("/")[-1])

    # Search and filters
    columns = st.columns(2)
    search_name = columns[0].text_input("Search for your model")
    if search_name:
        data = data[data["Model"].str.contains(search_name, case=False, na=False)]
    tags = columns[1].multiselect(
        "Open-source or closed-source models?",
        options=["Open", "Closed"],
        default=["Open", "Closed"],
    )
    if tags == ["Open"]:
        data = data[data["OpenSource"] != ""]
    if tags == ["Closed"]:
        data = data[data["OpenSource"] == ""]

    show_html_table(data)

    # st.dataframe(
    #     data[["Ranking", "Icon", "Model", "Organization", "Website", "ELO Score"]],
    #     hide_index=True,
    #     use_container_width=True,
    #     column_config={
    #         "Ranking": "",
    #         "Website": st.column_config.LinkColumn(display_text="Link"),
    #         "Icon": st.column_config.ImageColumn(label="", width="small"),
    #     },
    # )

    chart = (
        altair.Chart(data)
        .mark_bar()
        .encode(
            y=altair.Y(
                "Model",
                sort="-x",
                axis=altair.Axis(title=None, labelOverlap=False, labelLimit=200),
            ),
            x="ELO Score",
            color="ELO Score",
        )
    ).properties(height=30 * data.shape[0])
    st.altair_chart(chart, use_container_width=True)


def show_about():
    text = "This is a research work from ..."
    st.write(text)


def show_links():
    buttons_info = [
        dict(label="Blog", url="https://auto-arena.github.io/blog"),
        dict(label="Paper", url="https://arxiv.org/abs/2405.20267"),
        dict(label="Github", url="https://github.com/DAMO-NLP-SG/Auto-Arena-LLMs"),
    ]
    for i, column in enumerate(st.columns(len(buttons_info))):
        column.link_button(**buttons_info[i], use_container_width=True)


def show_logo(path: str):
    st.columns(9)[4].image(path)


def main(title: str = "🏆 Auto-Arena"):
    st.set_page_config(
        page_title=title, initial_sidebar_state="collapsed", layout="wide"
    )
    # st.header(title)
    # show_links()
    language = st.selectbox("Evaluation Language", ["English", "Chinese"])
    tabs = st.tabs(["Debate Samples", "Leaderboard Results"])
    intro = "We introduce the Auto-Arena, an automated and reliable framework for evaluating large language models in a human-like manner. We generate diverse and challenging questions across various domains, including writing, reasoning, and knowledge. During the debate battle stage, two models take turns answering questions and responding to each other's arguments, showcasing their deeper capabilities. Finally, a committee of the top-performing models is formed to review the debate process and judge the final winner through multi-round discussions. Below, we present our leaderboard of popular models:"
    folder = "data/main_tour_40" if language == "English" else "data/main_tour_40_zh"

    with tabs[0]:
        show_debates(folder)
    with tabs[1]:
        st.write(intro)
        show_results(folder)
        # show_logo("assets/logo.png")
        # components.iframe("https://auto-arena-leaderboard.hf.space", height=1000)
    # with tabs[2]:
    #     show_about()


if __name__ == "__main__":
    main()
