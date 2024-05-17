import json
from pathlib import Path
from typing import List

import altair
import pandas as pd
import streamlit as st


def get_organization(model_name: str):
    if "gpt" in model_name.lower():
        return "OpenAI"
    if "llama" in model_name.lower():
        return "Meta"
    if "glm" in model_name.lower():
        return "ZhipuAI"
    if "mistral" in model_name.lower() or "mixtral" in model_name.lower():
        return "Mistral"
    if "reka" in model_name.lower():
        return "Reka AI"
    if "claude" in model_name.lower():
        return "Anthropic"
    if "deepseek" in model_name.lower():
        return "Deepseek AI"
    if "qwen" in model_name.lower():
        return "Alibaba Cloud"
    if "command-r" in model_name.lower():
        return "Cohere"
    if "sensechat" in model_name.lower():
        return "SenseTime"
    if "wenxin" in model_name.lower():
        return "Baidu"
    if "zero-one-ai" in model_name.lower():
        return "01.AI"
    return ""


def show_debates(folder="../LLM_eval/data/main_tour_40"):
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

        for i, debate_round in enumerate(sample["rounds"]):
            st.subheader(f"Round {i + 1}")
            for j, (key, turn) in enumerate(debate_round):
                model = dict(a=candidates[0], b=candidates[1])[key]
                # st.subheader(f"Turn {j + 1}: {model}")
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
            # st.subheader(f"Judgement {k + 1}, Winner: {winner}")
            with st.chat_message("human", avatar=info_link):
                st.write(f"Judgement {k + 1}, Winner: {winner}")
            with st.chat_message("assistant"):
                st.write(text)


def show_results(path: str = "elo_history.csv"):
    data = pd.read_csv(path)
    data = data.transpose()
    data = data.iloc[1:]
    data = data.reset_index()
    data = data[["index", data.columns[-1]]]
    data.columns = ["Model", "ELO Score"]

    data[data.columns[-1]] = data[data.columns[-1]].round()
    data = data.sort_values(by=data.columns[-1], ascending=False)
    data.insert(0, "Ranking", [i + 1 for i in range(data.shape[0])])
    data.insert(1, "Organization", data["Model"].map(get_organization))
    st.dataframe(data, hide_index=True, use_container_width=True)

    # st.bar_chart(data, x="Model", y="ELO Score")
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
    )
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
    tabs = st.tabs(["Leaderboard Results", "Debate Samples", "About Us"])

    with tabs[0]:
        show_results()
    with tabs[1]:
        show_debates()
    with tabs[2]:
        show_about()


if __name__ == "__main__":
    main()
