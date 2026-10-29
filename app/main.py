import pandas as pd
import plotly.express as px
import streamlit as st

from database import Database
from experiment import Experiment
from clients import MambaClient, QwenClient


st.set_page_config(page_title="Mamba vs Qwen")
st.title("Mamba vs Qwen")
database = Database()
clients = {"Mamba": MambaClient(), "Qwen": QwenClient()}
health = {name: client.health() for name, client in clients.items()}
statuses = []
for name, status in health.items():
    state = "занята" if status.get("busy") else "готова"
    if status.get("status") != "ready":
        state = "недоступна"
    statuses.append(f"{name}: {state}")
st.caption(" · ".join(statuses))

choice = st.selectbox("Модель", ["Обе", "Mamba", "Qwen"])
selected_names = list(clients) if choice == "Обе" else [choice]
selected = [clients[name] for name in selected_names]
limits = {"10": 10, "100": 100, "1000": 1000, "Все": None}
limit = limits[st.selectbox("Количество примеров", list(limits))]
available = all(health[name].get("status") == "ready" for name in selected_names)
available = available and not any(status.get("busy") for status in health.values())
if st.button("Запустить", disabled=not available):
    try:
        results = Experiment(selected, database).run(limit)
        for name in selected_names:
            with st.spinner(f"Выполняется {name}…"):
                next(results)
    except Exception as error:
        st.error(f"Эксперимент остановлен: {error}. Завершённые измерения сохранены в истории.")

columns = ["model", "n_samples", "accuracy", "perplexity", "total_seconds", "seconds_per_sample"]
labels = {"model": "Модель", "n_samples": "Примеры", "accuracy": "Accuracy",
          "perplexity": "Perplexity", "total_seconds": "Время, с",
          "seconds_per_sample": "Секунд на пример", "run": "Запуск"}
history = database.load()
if history:
    frame = pd.DataFrame([result.values for result in history])
    frame["model"] = frame["model"].replace({client.name: name for name, client in clients.items()})
    latest = frame[frame.run_id == frame.run_id.iloc[0]]
    st.subheader("Последний запуск")
    st.dataframe(latest[columns].rename(columns=labels), hide_index=True)
    frame["time"] = pd.to_datetime(frame["started_at"], utc=True).dt.tz_convert("Europe/Moscow")
    frame["run"] = frame["time"].dt.strftime("%d.%m %H:%M:%S") + " · " + frame["run_id"].str[:8]
    with st.expander("История и фильтры"):
        model_filter = st.multiselect("Модели", frame.model.unique(), default=list(frame.model.unique()))
        sizes = sorted(frame.n_samples.unique())
        sample_filter = st.multiselect("Размер выборки", sizes, default=sizes)
        runs = list(frame.run.unique())
        run_filter = st.multiselect("Запуски", runs, default=runs[:5])
        shown = frame[frame.model.isin(model_filter) & frame.n_samples.isin(sample_filter) & frame.run.isin(run_filter)]
        st.dataframe(shown[["run"] + columns].rename(columns=labels), hide_index=True)
    for metric, title in [("accuracy", "Accuracy"), ("perplexity", "Perplexity"),
                          ("seconds_per_sample", "Секунд на пример")]:
        figure = px.bar(shown, x="run", y=metric, color="model", barmode="group",
                        title=title, labels=labels)
        st.plotly_chart(figure)
else:
    st.info("История пока пуста. Здесь появятся только результаты реальных измерений.")
st.caption("Perplexity оценивает последнее слово LAMBADA; разные токенизаторы ограничивают сравнение моделей.")
