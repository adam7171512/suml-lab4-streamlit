import streamlit as st
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextTask:
    title: str
    description: str
    model_id: str
    result_label: str
    label_map: dict[str, str]


SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "Panda0116/emotion-classification-model"

SENTIMENT_LABELS = {
    "POSITIVE": "Pozytywny",
    "NEGATIVE": "Negatywny",
}

EMOTION_LABELS = {
    "LABEL_0": "Smutek",
    "LABEL_1": "Radość",
    "LABEL_2": "Miłość",
    "LABEL_3": "Złość",
    "LABEL_4": "Strach",
    "LABEL_5": "Zaskoczenie",
    "sadness": "Smutek",
    "joy": "Radość",
    "love": "Miłość",
    "anger": "Złość",
    "fear": "Strach",
    "surprise": "Zaskoczenie",
}

TASKS = (
    TextTask(
        title="Wydźwięk emocjonalny tekstu",
        description="Klasyfikacja krótkiego tekstu po angielsku jako pozytywnego albo negatywnego.",
        model_id=SENTIMENT_MODEL,
        result_label="Wydźwięk",
        label_map=SENTIMENT_LABELS,
    ),
    TextTask(
        title="Wykrywanie emocji w tekście",
        description="Rozpoznawanie jednej z emocji: smutek, radość, miłość, złość, strach albo zaskoczenie.",
        model_id=EMOTION_MODEL,
        result_label="Emocja",
        label_map=EMOTION_LABELS,
    ),
)


@st.cache_resource
def load_text_classifier(model_id: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_id)


def show_text_prediction(task: TextTask) -> None:
    st.subheader(task.title)
    st.write(task.description)
    st.caption(f"Model Hugging Face: `{task.model_id}`")

    text = st.text_area(
        "Tekst do analizy",
        placeholder="Type English text to analyze...",
        height=160,
    )

    if not text.strip():
        st.info("Wpisz tekst po angielsku, aby zobaczyć wynik klasyfikacji.")
        return

    try:
        with st.spinner("Analizowanie tekstu..."):
            classifier = load_text_classifier(task.model_id)
            prediction = classifier(text, truncation=True)[0]
    except Exception as exc:
        st.error(f"Nie udało się wykonać predykcji: {exc}")
        return

    label = task.label_map.get(prediction["label"], prediction["label"].title())
    confidence = prediction["score"]

    st.divider()
    result_col, confidence_col = st.columns(2)
    result_col.metric(task.result_label, label)
    confidence_col.metric("Pewność", f"{confidence:.1%}")
    st.progress(confidence)


def main() -> None:
    st.set_page_config(
        page_title="Analiza tekstu",
        layout="centered",
    )

    st.title("Analiza tekstu z użyciem Hugging Face")
    st.write(
        "Aplikacja Streamlit pozwala wybrać jedną z dwóch funkcji NLP i uruchomić "
        "gotowy model klasyfikacji tekstu."
    )

    selected_task_title = st.selectbox(
        "Wybierz funkcję",
        [task.title for task in TASKS],
    )
    task = next(task for task in TASKS if task.title == selected_task_title)

    show_text_prediction(task)


def run() -> None:
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]

    from streamlit.web import cli as streamlit_cli

    streamlit_cli.main()


if __name__ == "__main__":
    main()
