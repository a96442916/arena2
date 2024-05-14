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


def main(path: str = "elo_history.csv"):
    st.header("Arena Leaderboard")

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


if __name__ == "__main__":
    main()
