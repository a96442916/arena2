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


def show_debates(folder: str):
    paths = sorted(Path(folder).glob("round*/*debate_history.jsonl"))
    debate_file = st.selectbox("Debate File", paths, format_func=lambda p: p.name)
    info_link = (
        "https://static-00.iconduck.com/assets.00/info-icon-2048x2048-tcgtx810.png"
    )

    with open(debate_file) as f:
        all_samples: List[dict] = [json.loads(line) for line in f]
        sample = st.selectbox(
            "Debate Question",
            options=all_samples,
            format_func=lambda s: f"{s['question']['domain'].capitalize()}: {s['question']['question']}",
        )
        candidates = sample["candidates"]
        with st.chat_message("human"):
            st.write(sample["question"]["question"])

        for i, debate_round in enumerate(sample["rounds"]):
            with st.expander(f"Round {i + 1}"):
                for j, (key, turn) in enumerate(debate_round):
                    model = dict(a=candidates[0], b=candidates[1])[key]
                    with st.chat_message("human", avatar=info_link):
                        st.write(f"Turn {j + 1}: {model}")
                    with st.chat_message("assistant"):
                        st.write(turn["original"])

    judgement_file = str(debate_file).replace("debate_history", "judge_results")
    with open(judgement_file) as f:
        all_judgements = [json.loads(line) for line in f]
        index = all_samples.index(sample)
        judge = st.selectbox("Judge", options=all_judgements[index]["judges"])
        for k, text in enumerate(all_judgements[index][judge]["judgement"]):
            key = all_judgements[index][judge]["winner"][k]
            winner = dict(A=candidates[0], B=candidates[1], tie="Tie")[key]
            with st.expander(f"Judgement {k + 1}, Winner: {winner}"):
                st.write(text)


def get_latest_elo_file(folder: str) -> str:
    num_rounds = len(sorted(Path(folder).glob("round*")))
    for p in sorted(Path(folder).iterdir()):
        if p.name.startswith(f"round{num_rounds}"):
            return str(Path(p, "elo_history.csv"))
    raise ValueError


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

    # For questions, show the full one without truncation also
    # For chat message style, show a left-right style between two bots
    # Support both english and chinese chats


if __name__ == "__main__":
    main()
