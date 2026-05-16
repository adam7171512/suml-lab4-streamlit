# Lab 4 - Streamlit

Prosta aplikacja Streamlit do analizy tekstu w języku angielskim.

W aplikacji można wybrać jedną z trzech funkcji:

1. sprawdzenie wydźwięku tekstu,
2. wykrywanie emocji w tekście,
3. tłumaczenie tekstu z języka angielskiego na język niemiecki.

Do predykcji użyte są gotowe modele z Hugging Face uruchamiane przez bibliotekę
`transformers`.

## Uruchomienie lokalne

```bash
uv run lab4
```

lub:

```bash
uv run streamlit run streamlit_app.py
```

Przy pierwszym uruchomieniu analiza może potrwać dłużej, ponieważ modele są
pobierane z Hugging Face.

Plik `requirements.txt` zawiera zależności potrzebne do uruchomienia aplikacji
po wdrożeniu na Streamlit Community Cloud.
