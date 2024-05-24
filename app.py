import base64
import json
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
        debates = {str(raw["gamekey"]): raw for raw in records}
    with open(debate_file.replace("debate_history", "judge_results")) as f:
        records = [json.loads(line) for line in f]
        judgements = {str(raw["gamekey"]): raw for raw in records}
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


def show_debates(folder: str):
    info = ModelsInfo()
    paths = sorted(Path(folder).glob("round*/*debate_history.jsonl"))
    columns = st.columns(2)
    model_a = columns[0].selectbox("Model A", find_models(paths, info))
    model_b = columns[1].selectbox("Model B", find_opponents(model_a, paths, info))
    debate_file = find_debate_file(model_a, model_b, paths)
    data = load_debates_and_judgements(str(debate_file))

    debate, judgements = st.selectbox(
        "Debate Question",
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

    judge = st.selectbox("Judge", options=judgements["judges"])
    mapping = dict(A=candidates[0], B=candidates[1], tie="Tie", error="Error")
    for k, text in enumerate(judgements[judge]["judgement"]):
        key = judgements[judge]["winner"][k]
        winner = mapping[key]
        with st.expander(f"Judgement {k + 1}, Winner: {winner}"):
            st.write(text)

    st.write(f"Overall winner: {mapping[judgements['final_winner'][-1]]}")


def get_latest_elo_file(folder: str) -> str:
    num_rounds = len(sorted(Path(folder).glob("round*")))
    for p in sorted(Path(folder).iterdir()):
        if p.name.startswith(f"round{num_rounds}"):
            return str(Path(p, "elo_history.csv"))
    raise ValueError


def show_html_table(data: pd.DataFrame):
    data = data.copy(deep=True)
    data["Organization"] = data.apply(
        lambda row: f'<img src="{row["Icon"]}" width="24"/> {row["Organization"]}',
        axis=1,
    )
    data["Model"] = data.apply(
        lambda row: f'<a href="{row["Website"]}">{row["Model"]}</a>',
        axis=1,
    )

    raw = data.to_html(
        columns=["Ranking", "Organization", "Model", "ELO Score"],
        justify="left",
        index=False,
        escape=False,
        render_links=True,
    )

    content = f"""
    <div style="height:400px; overflow:auto;">
        {raw}
    </div>
    """

    st.write(content, unsafe_allow_html=True)
    st.divider()


def show_results(folder: str):
    path = get_latest_elo_file(folder)
    data = pd.read_csv(path)
    data = data.transpose()
    data = data.iloc[1:]
    data = data.reset_index()
    data = data[["index", data.columns[-1]]]
    data.columns = ["Model", "ELO Score"]

    info = ModelsInfo()
    data[data.columns[-1]] = data[data.columns[-1]].round()
    data = data.sort_values(by=data.columns[-1], ascending=False)
    data.insert(0, "Ranking", [i + 1 for i in range(data.shape[0])])
    data.insert(1, "Organization", data["Model"].map(info.get_organization))
    data.insert(3, "Website", data["Model"].map(info.get_website))
    data.insert(1, "Icon", data["Model"].map(info.get_icon))

    st.dataframe(
        data,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Website": st.column_config.LinkColumn(display_text="Link"),
            "Icon": st.column_config.ImageColumn(width="small"),
        },
    )

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
    st.write("Placeholder: This is an LLM Arena Leaderboard from Alibaba")


def show_links():
    columns = st.columns(7)
    columns[0].link_button("Blog", url="")
    columns[1].link_button("Paper", url="")
    columns[2].link_button("Github", url="")
    columns[3].link_button("Dataset", url="")
    columns[4].link_button("Twitter", url="")
    columns[5].link_button("Discord", url="")


def main():
    st.header("🏆 Arena Leaderboard")
    show_links()
    language = st.selectbox("Evaluation Language", ["English", "Chinese"])
    folder = "data/main_tour_40" if language == "English" else "data/main_tour_40_zh"
    tabs = st.tabs(["Leaderboard Results", "Debate Samples", "About Us"])

    with tabs[0]:
        show_results(folder)
    with tabs[1]:
        show_debates(folder)
    with tabs[2]:
        show_about()


if __name__ == "__main__":
    main()
