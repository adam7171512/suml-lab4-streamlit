import streamlit as st
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextTask:
    title: str
    description: str
    model_id: str
    task_type: str
    result_label: str
    pipeline_task: str | None = None
    label_map: dict[str, str] | None = None


SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "Panda0116/emotion-classification-model"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-de"

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
        task_type="classification",
        result_label="Wydźwięk",
        pipeline_task="text-classification",
        label_map=SENTIMENT_LABELS,
    ),
    TextTask(
        title="Wykrywanie emocji w tekście",
        description="Rozpoznawanie jednej z emocji: smutek, radość, miłość, złość, strach albo zaskoczenie.",
        model_id=EMOTION_MODEL,
        task_type="classification",
        result_label="Emocja",
        pipeline_task="text-classification",
        label_map=EMOTION_LABELS,
    ),
    TextTask(
        title="Tłumaczenie z angielskiego na niemiecki",
        description="Tłumaczenie tekstu z języka angielskiego na język niemiecki.",
        model_id=TRANSLATION_MODEL,
        task_type="translation",
        result_label="Tłumaczenie",
    ),
)


@st.cache_resource
def load_text_pipeline(pipeline_task: str, model_id: str):
    from transformers import pipeline

    return pipeline(pipeline_task, model=model_id)


@st.cache_resource
def load_translation_model(model_id: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


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
        st.info("Wpisz tekst po angielsku, aby zobaczyć wynik.")
        return

    try:
        with st.spinner("Przetwarzanie tekstu..."):
            if task.task_type == "translation":
                import torch

                tokenizer, model = load_translation_model(task.model_id)
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                inputs = {name: value.to(model.device) for name, value in inputs.items()}
                with torch.no_grad():
                    output_tokens = model.generate(**inputs, max_length=512)
                translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            else:
                if task.pipeline_task is None:
                    raise ValueError("Brak typu pipeline dla zadania klasyfikacji.")
                text_pipeline = load_text_pipeline(task.pipeline_task, task.model_id)
                prediction = text_pipeline(text, truncation=True)[0]
    except Exception as exc:
        st.error(f"Nie udało się przetworzyć tekstu: {exc}")
        return

    if task.task_type == "translation":
        st.divider()
        st.markdown(f"**{task.result_label}:**")
        st.success(translation)
        return

    label_map = task.label_map or {}
    label = label_map.get(prediction["label"], prediction["label"].title())
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
        "Aplikacja Streamlit pozwala wybrać jedną z trzech funkcji NLP i uruchomić "
        "gotowy model z Hugging Face."
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
