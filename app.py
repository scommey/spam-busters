"""
SMS Spam Detection - Streamlit App
Spam Busters - MSc Business Analytics Text Analytics Project

Single self-contained file covering the full pipeline:
  data loading and quality checks -> regex cleaning -> text preprocessing ->
  exploratory text analytics -> TF-IDF -> SMOTE -> Naive Bayes + SVM ->
  evaluation, comparison and interpretation -> live prediction

Run locally:  streamlit run app.py
Every chart has a download button, so figures for the report can be saved
directly from the app.
"""

# ===============================================================
# 0. IMPORT PACKAGES / LIBRARIES
# ===============================================================
# Standard library: text handling, file buffers, regular expressions, counting
import base64
import html
import os
import io
import re
import ssl
from collections import Counter

# Data manipulation
import numpy as np
import pandas as pd

# Data visualisation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
from PIL import Image

# Web app
import streamlit as st

# Natural language processing (tokenisation, stopwords, lemmatisation, n-grams)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Text representation
from sklearn.feature_extraction.text import TfidfVectorizer

# Data splitting
from sklearn.model_selection import train_test_split

# Class imbalance handling
from imblearn.over_sampling import SMOTE

# Models: Naive Bayes and Support Vector Machine
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Model evaluation metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ===============================================================
# 1. SETTINGS
# ===============================================================
DATA_PATH = "SMSSpamCollection.tsv"         # dataset file (tab-separated: label, text)
RANDOM_STATE = 42                           # fixed seed so results are reproducible
LOGO_PATH = "spambusters_logo.png"          # Spam Busters company logo


# ===============================================================
# 2. LOAD DATA AND CHECK DATA QUALITY
# ===============================================================
def load_raw_data(path):
    # 2.1 Load and view the data
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")

    # 2.2 Data quality checks on the raw data
    #     (missing values, duplicates, inconsistent label values)
    quality = {
        "raw_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "label_values": sorted(df["label"].unique().tolist()),
    }

    # 2.3 Remove duplicate messages
    df = df.drop_duplicates().reset_index(drop=True)
    return df, quality


# ===============================================================
# 3. DATA CLEANING (REGULAR EXPRESSIONS)
# ===============================================================
def clean_text(text):
    text = html.unescape(text)                                       # decode entities: &lt;#&gt; -> <#>
    text = text.lower()                                              # case normalisation
    text = re.sub(r'<#>', ' num ', text)                             # dataset's masked-number placeholder
    text = re.sub(r'<url>', ' url ', text)                           # dataset's masked-URL placeholder
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)                # URLs -> placeholder
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', ' email ', text)  # emails -> placeholder
    text = re.sub(r'[£$€]', ' currency ', text)                      # currency symbols -> placeholder
    text = re.sub(r'\b\d{5,}\b', ' longnum ', text)                  # long digit strings -> placeholder
    text = re.sub(r'\d+', ' num ', text)                             # remaining digits -> placeholder
    text = re.sub(r'[^a-z\s]', ' ', text)                            # strip remaining punctuation
    text = re.sub(r'\s+', ' ', text).strip()                         # normalise whitespace
    return text


# ===============================================================
# 4. TEXT PREPROCESSING
# ===============================================================
# 4.1 Download the NLTK resources needed for preprocessing
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# 4.2 Tokenisation, stopword removal and lemmatisation
def preprocess_text(text):
    tokens = word_tokenize(text)                                          # tokenisation
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]    # stopword removal
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]                    # lemmatisation
    return " ".join(tokens)


# ===============================================================
# 5. APPLY CLEANING AND PREPROCESSING TO THE DATASET
# ===============================================================
def run_pipeline(path):
    # 5.1 Load the data and run the quality checks (section 2)
    df, quality = load_raw_data(path)

    # 5.2 Apply regex cleaning (section 3) and preprocessing (section 4)
    df["clean_text"] = df["text"].apply(clean_text)
    df["preprocessed_text"] = df["clean_text"].apply(preprocess_text)

    # 5.3 Message length features for exploratory analysis
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()
    return df, quality


# Cached so the data is processed once and reused by all pages
@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    return run_pipeline(DATA_PATH)


# ===============================================================
# 6. DATA SPLITTING, TEXT REPRESENTATION (TF-IDF),
#    CLASS IMBALANCE (SMOTE) AND MODEL BUILDING
# ===============================================================
# Cached so the models are trained once and reused by all pages
@st.cache_resource(show_spinner="Training models...")
def train_models(_df):
    # 6.1 Separate predictor (text) and target (label: ham = 0, spam = 1)
    X = _df["preprocessed_text"]
    y = _df["label"].map({"ham": 0, "spam": 1})

    # 6.2 Stratified 80/20 train-test split (keeps the ham/spam ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 6.3 TF-IDF text representation (fitted on training data only)
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 6.4 Class imbalance: SMOTE, applied after the split and after TF-IDF
    #     (SMOTE needs numeric vectors), to the training data only.
    #     The test set stays untouched.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

    # 6.5 Model 1: Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_res, y_train_res)

    # 6.6 Model 2: Linear Support Vector Machine
    svm = LinearSVC(random_state=RANDOM_STATE)
    svm.fit(X_train_res, y_train_res)

    return {
        "tfidf": tfidf,
        "nb": nb,
        "svm": svm,
        "X_test_tfidf": X_test_tfidf,
        "y_test": y_test,
        "train_counts_before": Counter(y_train),
        "train_counts_after": Counter(y_train_res),
    }


# ===============================================================
# 7. HELPER FUNCTIONS (evaluation, word counts, interpretation, charts)
# ===============================================================
# 7.1 Model evaluation metrics
def compute_metrics(y_test, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_score),
    }


# 7.2 Most frequent words and bigrams (exploratory analysis)
def top_n_words(texts, n=20):
    return Counter(" ".join(texts).split()).most_common(n)


def top_n_bigrams(texts, n=15):
    all_bigrams = []
    for t in texts:
        all_bigrams.extend(ngrams(t.split(), 2))
    return [(" ".join(bg), c) for bg, c in Counter(all_bigrams).most_common(n)]


# 7.3 Most influential terms per model (interpretation)
def top_features_nb(vectorizer, nb_model, n=15):
    """Terms with the largest log-probability difference between spam and ham."""
    names = np.array(vectorizer.get_feature_names_out())
    diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
    return names[np.argsort(diff)[-n:][::-1]].tolist(), names[np.argsort(diff)[:n]].tolist()


def top_features_svm(vectorizer, svm_model, n=15):
    """Terms with the largest positive (spam) and negative (ham) SVM weights."""
    names = np.array(vectorizer.get_feature_names_out())
    coef = svm_model.coef_[0]
    return names[np.argsort(coef)[-n:][::-1]].tolist(), names[np.argsort(coef)[:n]].tolist()


# 7.4 Chart display helpers
def show_figure(fig, filename):
    """Display a matplotlib figure with a PNG download button for the report."""
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button("Download PNG", buf.getvalue(), file_name=filename,
                       mime="image/png", key=f"dl_{filename}")


def bar_pair(data_pair, titles, figsize=(5, 5)):
    """Two horizontal bar charts (ham, spam) side by side, one per column."""
    cols = st.columns(2)
    figs = []
    for col, data, title, color in zip(cols, data_pair, titles, [HAM_COLOR, SPAM_COLOR]):
        with col:
            fig, ax = plt.subplots(figsize=figsize)
            words, counts = zip(*sorted(data, key=lambda x: x[1]))
            ax.barh(words, counts, color=color)
            ax.grid(axis="y", visible=False)
            ax.set_title(title)
            plt.tight_layout()
            figs.append((col, fig, title))
    return figs


# 7.5 Layout helpers (page headings, section headings, cards, SMS bubbles)
def page_header(title, intro):
    st.markdown(
        f'<div class="page-head"><h2>{title}</h2><p>{intro}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, note=""):
    note_html = f"<p>{note}</p>" if note else ""
    st.markdown(f'<div class="sec"><h4>{title}</h4>{note_html}</div>', unsafe_allow_html=True)


def stat_cards(cards):
    """cards: list of (label, value, sub_text, tone) where tone is '', 'ham' or 'spam'."""
    html_cards = "".join(
        f'<div class="stat {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub, tone in cards
    )
    st.markdown(f'<div class="stats">{html_cards}</div>', unsafe_allow_html=True)


def bubble(text, tone, tag):
    return (f'<div class="bubble {tone}"><span class="tag {tone}">{tag}</span><br>'
            f'{html.escape(text)}</div>')


# ===============================================================
# 8. APP STYLING (colours, chart style, page design)
# ===============================================================
# 8.1 Colour palette (ham = teal, spam = red, used in every chart and card)
INK = "#1D2B3A"        # text and sidebar
PAPER = "#F3F6F9"      # page background
LINE = "#D9E1E8"       # borders
HAM_COLOR = "#1F8A70"
SPAM_COLOR = "#D1495B"
SVM_COLOR = "#2F5D8A"
NB_COLOR = "#E0A030"

# 8.2 Chart style (applies to every matplotlib figure)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#EEF2F6",
    "xtick.color": "#5B6B7C",
    "ytick.color": "#5B6B7C",
    "legend.frameon": False,
    "font.size": 10,
})
BLUE_MAP = LinearSegmentedColormap.from_list("blue", ["#FFFFFF", SVM_COLOR])
HAM_MAP = LinearSegmentedColormap.from_list("ham", ["#0B3D30", HAM_COLOR, "#5CC2A6"])
SPAM_MAP = LinearSegmentedColormap.from_list("spam", ["#6B1422", SPAM_COLOR, "#F08A97"])

# 8.3 Page styling (fonts, colours, layout)
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button {{
    font-family: 'IBM Plex Sans', 'Source Sans', 'Source Sans Pro', 'Segoe UI', sans-serif;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
h1, h2, h3, h4, .brand, .hero h1 {{ font-family: 'Bricolage Grotesque', 'Source Sans', 'Source Sans Pro', 'Segoe UI', sans-serif; color: {INK}; }}
.block-container {{ padding-top: 4.5rem; max-width: 1180px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {INK}; }}
section[data-testid="stSidebar"] * {{ color: #DCE4EC; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.45rem 0.6rem; border-radius: 8px; margin-bottom: 2px; width: 100%;
}}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.05); }}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    box-shadow: inset 3px 0 0 {HAM_COLOR};
    background: rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
    color: #FFFFFF; font-weight: 600;
}}
.brand {{ font-size: 1.3rem !important; font-weight: 700; color: #FFFFFF !important; margin: 0; }}
.brand-sub {{ font-size: 0.85rem !important; color: #9FB0C2 !important; margin: 0.2rem 0 1.4rem 0; }}

/* Spam Busters logo */
.side-logo {{ background: #FFFFFF; border-radius: 14px; padding: 0.7rem; margin-bottom: 1rem; }}
.side-logo img {{ width: 100%; display: block; }}
.mobile-logo {{ display: none; }}
.mobile-logo img {{ width: 9rem; display: block; border-radius: 12px; }}

/* Hero (page 1) */
.hero {{
    background: {INK}; border-radius: 20px; padding: 2.4rem 2.6rem;
    display: grid; grid-template-columns: 1.15fr 1fr; gap: 2rem; align-items: center;
    margin-bottom: 1.8rem;
}}
.hero h1 {{ color: #FFFFFF; font-size: 2.6rem; line-height: 1.08; margin: 0 0 0.8rem 0; padding: 0; }}
.hero p {{ color: #B9C6D3; font-size: 1.02rem; line-height: 1.6; margin: 0; max-width: 34rem; }}
.thread {{ display: flex; flex-direction: column; gap: 0.9rem; }}

/* SMS bubbles */
.bubble {{
    max-width: 92%; padding: 0.8rem 1rem; border-radius: 18px 18px 18px 4px;
    font-size: 0.95rem; line-height: 1.45; position: relative;
}}
.bubble.ham {{ background: #FFFFFF; color: {INK}; border: 1px solid {LINE}; }}
.bubble.spam {{ background: #FBE9EC; color: #4A1520; border: 1px solid #F3C4CC; }}
.tag {{
    display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.12rem 0.55rem;
    border-radius: 999px; margin-bottom: 0.4rem; color: #FFFFFF;
}}
.tag.ham {{ background: {HAM_COLOR}; }}
.tag.spam {{ background: {SPAM_COLOR}; }}

/* Page and section headings */
.page-head {{ margin: 0.2rem 0 1.4rem 0; }}
.page-head h2 {{ font-size: 2rem; margin: 0.1rem 0 0.35rem 0; padding: 0; }}
.page-head p {{ color: #4A5A6B; margin: 0; max-width: 46rem; }}
.sec {{ border-top: 1px solid {LINE}; margin: 1.8rem 0 0.8rem 0; padding-top: 1rem; }}
.sec h4 {{ font-size: 1.15rem; margin: 0; padding: 0; }}
.sec p {{ color: #5B6B7C; font-size: 0.9rem; margin: 0.2rem 0 0 0; }}

/* Stat cards */
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; }}
.stat {{
    background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
    padding: 0.9rem 1.1rem; border-top: 4px solid {INK};
}}
.stat.ham {{ border-top-color: {HAM_COLOR}; }}
.stat.spam {{ border-top-color: {SPAM_COLOR}; }}
.stat .label {{ font-size: 0.85rem; color: #5B6B7C; }}
.stat .value {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.9rem; font-weight: 700; }}
.stat .sub {{ font-size: 0.82rem; color: #5B6B7C; }}

/* Preprocessing stages */
.stage {{
    display: grid; grid-template-columns: 2.2rem 1fr; gap: 0.8rem;
    background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
    padding: 0.85rem 1rem; margin-bottom: 0.6rem;
}}
.stage .num {{
    width: 2.2rem; height: 2.2rem; border-radius: 50%; background: {INK}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; font-weight: 600;
}}
.stage .name {{ font-weight: 600; font-size: 0.9rem; }}
.stage .text {{ font-size: 0.95rem; color: #33475B; word-break: break-word; }}

/* Prediction verdict */
.verdict {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 16px; padding: 1.3rem 1.4rem; }}
.verdict .result {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 0.6rem 0 0.2rem 0; }}
.verdict .result.ham {{ color: {HAM_COLOR}; }}
.verdict .result.spam {{ color: {SPAM_COLOR}; }}
.verdict .note {{ color: #5B6B7C; font-size: 0.88rem; }}

/* Widgets */
.stButton button[kind="primary"] {{ background: {INK}; border-color: {INK}; }}
.stButton button[kind="primary"]:hover {{ background: {SVM_COLOR}; border-color: {SVM_COLOR}; }}
.stDownloadButton button {{ font-size: 0.8rem; padding: 0.2rem 0.7rem; min-height: 0; }}
[data-testid="stExpander"] {{ background: #FFFFFF; border-radius: 12px; }}
.stTextArea textarea {{ background: #FFFFFF; border: 1px solid {LINE}; font-size: 1rem; }}
button:focus-visible {{ outline: 2px solid {SVM_COLOR}; outline-offset: 2px; }}

@media (max-width: 800px) {{
    .mobile-logo {{ display: block; margin: -1.5rem 0 1.2rem 0; }}
    .hero {{ grid-template-columns: 1fr; padding: 1.6rem; }}
    .hero h1 {{ font-size: 2rem; }}
}}
</style>
"""

# 8.4 Spam Busters logo (embedded as an image in the page design)
@st.cache_data
def logo_data_uri():
    """Return the logo as a data URI for use in HTML, or "" if the file is missing."""
    if not os.path.exists(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def logo_html(css_class):
    uri = logo_data_uri()
    return f'<div class="{css_class}"><img src="{uri}" alt="Spam Busters logo"></div>' if uri else ""


def page_icon():
    """Shield part of the logo for the browser tab; falls back to an emoji."""
    if not os.path.exists(LOGO_PATH):
        return "📱"
    logo = Image.open(LOGO_PATH)
    w, h = logo.size
    return logo.crop((int(w * 0.1), 0, int(w * 0.8), int(h * 0.7)))


# 8.5 Apply the page styling (called at the start of the app, section 14)
def apply_styling():
    st.markdown(CSS, unsafe_allow_html=True)


# ===============================================================
# 9. APP PAGE - ABOUT THE DATA (dataset overview + quality checks)
# ===============================================================
def page_overview(df, quality):
    page_header("About the data", "What is in the dataset and how clean it was before any processing.")

    # 9.1 Dataset size and class counts
    n_ham = int((df["label"] == "ham").sum())
    n_spam = int((df["label"] == "spam").sum())
    stat_cards([
        ("Messages after cleaning", f"{len(df):,}", f"from {quality['raw_rows']:,} loaded", ""),
        ("Ham", f"{n_ham:,}", f"{n_ham / len(df):.1%} of messages", "ham"),
        ("Spam", f"{n_spam:,}", f"{n_spam / len(df):.1%} of messages", "spam"),
    ])

    # 9.2 Data quality checks table
    section("Data quality checks", "Run on the raw file, before cleaning.")
    missing_total = sum(quality["missing_values"].values())
    checks = pd.DataFrame(
        {
            "Check": ["Rows loaded", "Missing values", "Duplicate rows removed",
                      "Rows after cleaning", "Label values"],
            "Result": [
                str(quality["raw_rows"]),
                "None" if missing_total == 0 else str(quality["missing_values"]),
                str(quality["duplicates"]),
                str(len(df)),
                ", ".join(quality["label_values"]),
            ],
        }
    )
    st.dataframe(checks, hide_index=True, width="content")

    # 9.3 Optional views of the raw and preprocessed data
    section("Browse the data")
    if st.checkbox("Show raw dataset", key="show_raw_dataset"):
        st.dataframe(df[["label", "text"]], width="stretch")

    if st.checkbox("Show preprocessed dataset", key="show_preprocessed_dataset"):
        st.dataframe(df[["label", "text", "clean_text", "preprocessed_text"]], width="stretch")


# ===============================================================
# 10. APP PAGE - HOW MESSAGES ARE CLEANED (text preprocessing, one message at each stage)
# ===============================================================
def page_preprocessing(df):
    page_header("How messages are cleaned",
                "Pick any message to see how it changes at each stage of cleaning.")

    # 10.1 Select a message and show it before and after each step
    sample_idx = st.slider("Message number", 0, len(df) - 1, 0, key="sample_idx")
    row = df.iloc[sample_idx]

    tone = row["label"]
    st.markdown(f'<span class="tag {tone}">{tone}</span>', unsafe_allow_html=True)
    stages = [
        ("Original message", row["text"]),
        ("After regex cleaning", row["clean_text"]),
        ("After tokenisation, stopword removal and lemmatisation", row["preprocessed_text"]),
    ]
    for i, (name, text) in enumerate(stages, 1):
        st.markdown(
            f'<div class="stage"><div class="num">{i}</div><div>'
            f'<div class="name">{name}</div><div class="text">{html.escape(text) or "(empty)"}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ===============================================================
# 11. APP PAGE - WHAT SPAM LOOKS LIKE (exploratory text analytics)
# ===============================================================
def page_eda(df):
    page_header("What spam looks like",
                "How ham and spam messages differ in balance, length, vocabulary and style.")

    ham = df[df["label"] == "ham"]
    spam = df[df["label"] == "spam"]

    # 11.1 Class distribution and imbalance level
    section("Class distribution")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df["label"].value_counts()
        counts.plot(kind="bar", ax=ax, color=[HAM_COLOR, SPAM_COLOR], rot=0)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom")
        ax.grid(axis="x", visible=False)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of messages")
        plt.tight_layout()
        show_figure(fig, "eda_class_distribution.png")
    with col2:
        pct = df["label"].value_counts(normalize=True)
        stat_cards([
            ("Ham", f"{pct['ham']:.1%}", "of messages", "ham"),
            ("Spam", f"{pct['spam']:.1%}", "of messages", "spam"),
        ])
        st.write("")
        st.info(
            f"The ratio of roughly {pct['ham']:.0%}/{pct['spam']:.0%} falls within the "
            "70/30 to 90/10 range, a **moderate class imbalance**. Accuracy alone would "
            "be misleading, so models are evaluated with precision, recall, F1 and "
            "ROC-AUC, the split is stratified, and SMOTE is applied to the training data."
        )

    # 11.2 Message length: summary statistics and histograms
    section("Message length", "Spam messages tend to be longer than ham messages.")
    length_stats = df.groupby("label")[["char_len", "word_len"]].agg(["mean", "median", "min", "max"]).round(1)
    length_stats.columns = [f"{c} ({s})" for c, s in length_stats.columns]
    st.dataframe(length_stats, width="stretch")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, color in zip(["ham", "spam"], [HAM_COLOR, SPAM_COLOR]):
        subset = df[df["label"] == label]
        axes[0].hist(subset["char_len"], bins=40, alpha=0.7, label=label, color=color)
        axes[1].hist(subset["word_len"], bins=25, alpha=0.7, label=label, color=color)
    axes[0].set_title("Message Length (characters)")
    axes[0].set_xlabel("Characters")
    axes[0].legend()
    axes[1].set_title("Message Length (words)")
    axes[1].set_xlabel("Words")
    axes[1].legend()
    plt.tight_layout()
    show_figure(fig, "eda_message_length.png")

    # 11.3 Top 20 words by class
    section("Most frequent words", "Top 20 words in each class after preprocessing.")
    for col, fig, title in bar_pair(
        [top_n_words(ham["preprocessed_text"], 20), top_n_words(spam["preprocessed_text"], 20)],
        ["Top 20 words - HAM", "Top 20 words - SPAM"],
    ):
        with col:
            show_figure(fig, f"eda_{title.split()[-1].lower()}_top_words.png")

    # 11.4 Top 15 bigrams by class
    section("Most frequent word pairs (bigrams)", "Top 15 two-word combinations in each class.")
    for col, fig, title in bar_pair(
        [top_n_bigrams(ham["preprocessed_text"], 15), top_n_bigrams(spam["preprocessed_text"], 15)],
        ["Top 15 bigrams - HAM", "Top 15 bigrams - SPAM"],
    ):
        with col:
            show_figure(fig, f"eda_{title.split()[-1].lower()}_top_bigrams.png")

    # 11.5 Word clouds by class
    section("Word clouds")
    col1, col2 = st.columns(2)
    for col, subset, title, cmap in zip([col1, col2], [ham, spam], ["HAM", "SPAM"], [HAM_MAP, SPAM_MAP]):
        with col:
            wc = WordCloud(width=500, height=350, background_color="white",
                           colormap=cmap).generate(" ".join(subset["preprocessed_text"]))
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title)
            show_figure(fig, f"eda_{title.lower()}_wordcloud.png")

    # 11.6 Surface signals: digits, capitals, exclamation marks, currency
    section("Surface signals", "Average use of digits, capitals, exclamation marks and currency symbols.")

    def digit_ratio(t):
        return sum(c.isdigit() for c in t) / max(len(t), 1)

    def upper_word_count(t):
        return sum(1 for w in t.split() if w.isupper() and len(w) > 1)

    def has_currency(t):
        return bool(re.search(r"[£$€]", t))

    signals = pd.DataFrame({
        "label": df["label"],
        "digit_ratio": df["text"].apply(digit_ratio),
        "upper_word_count": df["text"].apply(upper_word_count),
        "exclaim_count": df["text"].str.count("!"),
        "has_currency": df["text"].apply(has_currency),
    })
    summary = signals.groupby("label").agg(
        avg_digit_ratio=("digit_ratio", "mean"),
        avg_upper_words=("upper_word_count", "mean"),
        avg_exclaim_count=("exclaim_count", "mean"),
        pct_with_currency=("has_currency", "mean"),
    ).round(3)
    st.dataframe(summary, width="stretch")


# ===============================================================
# 12. APP PAGE - MODEL PERFORMANCE (TF-IDF, evaluation, comparison and interpretation)
# ===============================================================
def page_models(df):
    page_header("Model performance",
                "Both models are trained on TF-IDF features and tested on the same held-out 20% of messages.")

    # 12.1 Load the trained models (section 6)
    trained = train_models(df)
    tfidf = trained["tfidf"]
    nb, svm = trained["nb"], trained["svm"]
    X_test_tfidf, y_test = trained["X_test_tfidf"], trained["y_test"]

    # 12.2 TF-IDF size and class counts before and after SMOTE
    before, after = trained["train_counts_before"], trained["train_counts_after"]
    stat_cards([
        ("TF-IDF vocabulary", f"{len(tfidf.get_feature_names_out()):,}", "unigrams and bigrams", ""),
        ("Training spam before SMOTE", f"{before[1]:,}", f"vs {before[0]:,} ham", "spam"),
        ("Training spam after SMOTE", f"{after[1]:,}", f"vs {after[0]:,} ham", "spam"),
        ("Test messages", f"{len(y_test):,}", "not resampled", ""),
    ])
    st.caption("SMOTE is applied to the training data only, so the test set keeps the real ham/spam ratio.")

    # 12.3 Predict on the untouched test set
    nb_pred = nb.predict(X_test_tfidf)
    nb_score = nb.predict_proba(X_test_tfidf)[:, 1]
    svm_pred = svm.predict(X_test_tfidf)
    svm_score = svm.decision_function(X_test_tfidf)

    # 12.4 Evaluation metrics and model comparison
    section("Model evaluation on the test set")
    results_df = pd.DataFrame({
        "Naive Bayes": compute_metrics(y_test, nb_pred, nb_score),
        "SVM": compute_metrics(y_test, svm_pred, svm_score),
    }).T
    st.dataframe(results_df.round(4), width="stretch")
    st.download_button("Download CSV", results_df.round(4).to_csv().encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

    # 12.5 Best model by F1-score
    best = results_df["F1-score"].idxmax()
    st.success(f"Best model by F1-score: **{best}** ({results_df.loc[best, 'F1-score']:.4f})")

    # 12.6 Per-class classification reports
    with st.expander("Classification reports (per class)"):
        col1, col2 = st.columns(2)
        for col, pred, name in zip([col1, col2], [nb_pred, svm_pred], ["Naive Bayes", "SVM"]):
            with col:
                st.markdown(f"**{name}**")
                report = classification_report(y_test, pred, target_names=["ham", "spam"],
                                               output_dict=True)
                st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    # 12.7 Confusion matrices and ROC curves
    section("Confusion matrices and ROC curves")
    col1, col2 = st.columns(2)
    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, pred, title in zip(axes, [nb_pred, svm_pred], ["Naive Bayes", "SVM"]):
            cm = confusion_matrix(y_test, pred)
            ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(
                ax=ax, colorbar=False, cmap=BLUE_MAP
            )
            ax.grid(False)
            ax.set_title(title)
        plt.tight_layout()
        show_figure(fig, "model_confusion_matrices.png")
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for score, name, color in [(nb_score, "Naive Bayes", NB_COLOR), (svm_score, "SVM", SVM_COLOR)]:
            fpr, tpr, _ = roc_curve(y_test, score)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC={roc_auc_score(y_test, score):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA8B6", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        plt.tight_layout()
        show_figure(fig, "model_roc_curves.png")

    # 12.8 Interpretation: most influential terms per model
    section("Interpretation: most influential terms",
            "The words that push each model most strongly towards spam or ham.")
    nb_spam, nb_ham = top_features_nb(tfidf, nb)
    svm_spam, svm_ham = top_features_svm(tfidf, svm)
    st.dataframe(
        pd.DataFrame({
            "SVM: spam terms": svm_spam,
            "SVM: ham terms": svm_ham,
            "Naive Bayes: spam terms": nb_spam,
            "Naive Bayes: ham terms": nb_ham,
        }),
        width="stretch",
    )
    st.caption(
        "SVM terms are ranked by model weight; Naive Bayes terms by the difference in "
        "log-probability between the spam and ham classes."
    )


# ===============================================================
# 13. APP PAGE - CHECK A MESSAGE (live prediction, opening page)
# ===============================================================
# Reset the text box when "Clear" is clicked
def clear_text():
    st.session_state.user_sms = ""


def page_predict(df):
    # 13.1 Hero: what the app does, with a real ham and spam message from the dataset
    ham_example = df[df["label"] == "ham"]["text"].iloc[1]
    spam_example = df[df["label"] == "spam"]["text"].iloc[0]
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <h1>Is this message spam?</h1>
            <p>Paste any text message below to check whether it is spam or a
            legitimate message (ham). The detector was trained on {len(df):,} real
            messages from the UCI SMS Spam Collection.</p>
          </div>
          <div class="thread">
            {bubble(ham_example, "ham", "ham")}
            {bubble(spam_example[:140].rsplit(" ", 1)[0] + " …", "spam", "spam")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trained = train_models(df)
    tfidf, nb, svm = trained["tfidf"], trained["nb"], trained["svm"]

    # 13.2 Choose a model and enter a message
    model_choice = st.radio("Choose model", ["SVM (recommended)", "Naive Bayes"], horizontal=True)

    if "user_sms" not in st.session_state:
        st.session_state.user_sms = ""

    user_input = st.text_area("Enter an SMS message", key="user_sms", height=120)

    col1, col2, _ = st.columns([1, 1, 4])
    predict_clicked = col1.button("Predict", type="primary", width="stretch")
    col2.button("Clear", width="stretch", on_click=clear_text)

    # 13.3 Clean, preprocess and vectorise the message, then predict
    if predict_clicked:
        if not user_input.strip():
            st.warning("Enter a message first, then click Predict.")
        else:
            processed = preprocess_text(clean_text(user_input))
            vec = tfidf.transform([processed])

            if model_choice.startswith("SVM"):
                pred = svm.predict(vec)[0]
                score = svm.decision_function(vec)[0]
                confidence_note = f"SVM decision score: {score:.2f} (further from 0 = more confident)"
            else:
                pred = nb.predict(vec)[0]
                proba = nb.predict_proba(vec)[0][1]
                confidence_note = f"Naive Bayes spam probability: {proba:.2%}"

            # 13.4 Show the message as an SMS bubble with the verdict
            tone = "spam" if pred == 1 else "ham"
            verdict = "This message is spam" if pred == 1 else "This message is legitimate (ham)"
            st.markdown(
                f'<div class="verdict">{bubble(user_input, tone, tone)}'
                f'<div class="result {tone}">{verdict}</div>'
                f'<div class="note">{confidence_note}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Preprocessed text used for prediction: `{processed}`")


# ===============================================================
# 14. APP NAVIGATION (sidebar menu and run the app)
# ===============================================================
def main():
    # 14.1 Page setup (must be the first Streamlit command), apply the styling
    #      (section 8) and load the data (sections 2-5)
    st.set_page_config(page_title="SMS Spam Detection | Spam Busters", page_icon=page_icon(), layout="wide")
    apply_styling()
    df, quality = load_data()

    # 14.2 Phone-only logo at the top of every page (the sidebar is hidden on phones)
    st.markdown(logo_html("mobile-logo"), unsafe_allow_html=True)

    # 14.3 Sidebar: Spam Busters logo, app name and page menu
    st.sidebar.markdown(
        logo_html("side-logo")
        + '<p class="brand">SMS Spam Detection</p>'
        '<p class="brand-sub">Spam Busters, MSc Business Analytics</p>',
        unsafe_allow_html=True,
    )
    pages = {
        "Check a message": lambda: page_predict(df),
        "About the data": lambda: page_overview(df, quality),
        "How messages are cleaned": lambda: page_preprocessing(df),
        "What spam looks like": lambda: page_eda(df),
        "Model performance": lambda: page_models(df),
    }
    selection = st.sidebar.radio("Pages", list(pages.keys()), label_visibility="collapsed")

    # 14.4 Show the selected page
    pages[selection]()


if __name__ == "__main__":
    main()
