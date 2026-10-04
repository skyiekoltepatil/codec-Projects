# codec-Projects — AI/ML Internship Project Suite

Ten independently runnable machine-learning applications, each with real data,
real training, honest evaluation and a consistent Streamlit interface. A single
root dashboard (`app.py`) lists every project and launches the one you pick.

- **Language / runtime:** Python 3.11+
- **Classical ML:** scikit-learn, pandas, NumPy, SciPy
- **Deep learning:** PyTorch (CPU, Apple Metal/MPS and CUDA all supported)
- **Interfaces:** Streamlit, Plotly, Matplotlib, Seaborn
- **Speech:** faster-whisper (offline Whisper transcription)
- **Testing:** pytest (one process per project — see *Testing* below)

The guiding principle of the repository is **honesty over polish**: every number
reported is produced by the code in this repository on a held-out split, weak
results are labelled as weak, dataset substitutions are documented rather than
hidden, and forecasting projects are explicitly marked as educational rather
than production-grade.

---

## Table of contents

- [What this repository is](#what-this-repository-is)
- [Repository layout](#repository-layout)
- [The ten projects](#the-ten-projects)
- [Architecture: how the pieces fit together](#architecture-how-the-pieces-fit-together)
- [Installation](#installation)
- [Running the projects](#running-the-projects)
- [Training the models](#training-the-models)
- [Testing](#testing)
- [Datasets and licensing](#datasets-and-licensing)
- [Engineering challenges and how they were solved](#engineering-challenges-and-how-they-were-solved)
- [Known limitations](#known-limitations)
- [License](#license)

---

## What this repository is

This is a portfolio-grade collection of ten small but complete ML products. Each
one is a self-contained folder with the same internal shape:

```
NN-project-slug/
├── app.py            # Streamlit application (the product)
├── train.py          # training / evaluation entry point (when trainable)
├── src/              # the project's own package: data, features, model
├── models/           # trained artefacts (created by train.py, git-ignored)
├── data/             # downloaded / cached datasets (git-ignored)
├── notebooks/        # exploratory analysis
├── screenshots/      # interface captures
└── .streamlit/config.toml   # per-project theme
```

Nothing heavy is committed: datasets download on demand into each project's
`data/` directory and models are trained locally into `models/`. A fresh clone
contains only source code, tiny text fixtures and configuration.

Teamed with the ten projects is a small shared library (`shared/`) that gives
every app the same data-acquisition layer, evaluation vocabulary, theming and
error handling. That shared code is where most of the engineering lessons in
this project live — see
[Engineering challenges and how they were solved](#engineering-challenges-and-how-they-were-solved).

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Root dashboard — lists all ten projects, launches them, shows readiness. |
| `shared/` | Shared library used by every project (see the architecture section). |
| `01-…` … `10-…` | The ten independent projects. |
| `scripts/setup.sh` | Create the `.venv` and install dependencies. |
| `scripts/train_all.sh` | Train every trainable project in sequence. |
| `scripts/run_tests.sh` | Run the whole test suite (one pytest process per module). |
| `tests/` | Test suite plus the project-import isolation helper. |
| `requirements.txt` | Pinned dependency set (`>=` lower bound plus a major ceiling). |
| `pyproject.toml` | Package metadata, pytest and Ruff configuration. |

## The ten projects

| # | Project | ML type | Core algorithms | Dataset |
| --- | --- | --- | --- | --- |
| 01 | [Stock Price Predictor](01-stock-price-predictor) | Regression / time series | Linear Regression, Random Forest, Gradient Boosting + persistence baseline | Daily closes for AAPL, MSFT, IBM, SBUX, S&P 500 (2007–2016) |
| 02 | [Twitter/X Sentiment Analysis](02-twitter-sentiment-analysis) | NLP / classification | TF-IDF + Logistic Regression, Naive Bayes, Linear SVM | Twitter US Airline Sentiment (14,196 labelled tweets) |
| 03 | [Handwritten Digit Recogniser](03-handwritten-digit-recognizer) | Vision / deep learning | CNN (two conv blocks, batch norm, dropout) | MNIST (70,000 28×28 images) |
| 04 | [Movie Recommendation System](04-movie-recommendation-system) | Recommender system | TF-IDF content similarity + truncated-SVD collaborative filtering | MovieLens `ml-latest-small` (100k ratings, 9,742 films) |
| 05 | [Customer Churn Prediction](05-customer-churn-prediction) | Classification | Logistic Regression, Random Forest, Gradient Boosting | IBM Telco Customer Churn (7,043 customers) |
| 06 | [Customer Service Chatbot](06-customer-service-chatbot) | NLP / intent classification | TF-IDF + Logistic Regression with a confidence gate | Curated intents corpus (8 intents, hand-written) |
| 07 | [Spam Email Classifier](07-spam-email-classifier) | NLP / binary classification | TF-IDF + Multinomial Naive Bayes, Logistic Regression, Linear SVM | UCI SMS Spam Collection (5,574 messages) |
| 08 | [Fruit Image Classifier](08-fruit-image-classifier) | Vision / deep learning | CNN (or MobileNetV2 transfer learning) with augmentation | Fruits-360 → 7 fruits, 5,150 images |
| 09 | [Weather Analysis & Prediction](09-weather-analysis-prediction) | Regression / time series | Linear Regression, Random Forest, Gradient Boosting | Open-Meteo historical daily archive |
| 10 | [Speech-to-Text Transcription](10-speech-to-text) | Speech / deep learning | OpenAI Whisper via faster-whisper (int8, offline) | No training data — pretrained weights on first use |

Each project's interface reports the same categories of information: the model
used, the data it was trained on, the held-out metrics, and any caveat that
applies to the result. Models that are *not* better than a trivial baseline (for
example, the persistence baseline in project 01) are reported as such rather than
presented as skillful.

## Architecture: how the pieces fit together

Consistency comes from a single shared library. Each module solves a recurring
problem once so that no individual project has to:

| Module | Responsibility |
| --- | --- |
| `shared/projects.py` | Single source of truth for project metadata (names, algorithms, datasets). The dashboard and any README tables cannot disagree because they all read from here. |
| `shared/paths.py` | Platform-independent path resolution, all derived from the repository location. |
| `shared/config.py` | Logging setup and optional environment variables (every one has a working default). |
| `shared/errors.py` | Typed, user-facing exceptions (`ProjectError` and subclasses) rendered as actionable messages instead of raw tracebacks. |
| `shared/datasets.py` | Atomic, checksum-verified dataset downloads with mirrors, retries and safe archive extraction. |
| `shared/ml.py` | Reproducible seeding, device selection (CPU/MPS/CUDA), artefact I/O, leakage-free splitting and a timing helper. |
| `shared/metrics.py` | Regression, classification and ranking metrics in one JSON-serialisable shape. |
| `shared/text.py` | Text cleaning, tokenisation and stop-word handling shared by the three NLP projects. |
| `shared/plotting.py` | Shared chart builders so every project looks the same. |
| `shared/theme.py` | One restrained visual language (slate palette, generous whitespace) for all apps. |
| `shared/ui.py` | Streamlit rendering helpers (errors, figures, tables, metric cards, downloads). |
| `shared/launcher.py` | Starts and stops each project's Streamlit app as an isolated child process. |

The root `app.py` reads `shared/projects.py`, shows a readiness check for each
project (is a trained artefact present? is it a pretrained model?), and starts the
selected project on its own port via `shared/launcher.py`.

Why a launcher instead of importing each app directly is itself one of the core
engineering challenges, documented [below](#1-the-src-package-name-collision).

## Installation

Requires **Python 3.11 or newer**.

```bash
git clone https://github.com/skyiekoltepatil/codec-Projects.git
cd codec-Projects

# Idempotent: creates .venv, upgrades pip, installs requirements.txt
./scripts/setup.sh

# Activate it
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Or install manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency versions are pinned as a lower bound plus a **major-version ceiling**
(for example `scikit-learn>=1.5,<2.0`) so that a future minor release cannot
silently change model behaviour. All versions were verified on Python 3.11 on
macOS (Apple Silicon) and install on Linux and Windows. PyTorch ships a native
arm64 wheel, so CPU and Apple Metal work without a separate framework install;
CUDA is used automatically when present and is never required.

## Running the projects

### The dashboard (all ten in one place)

```bash
streamlit run app.py
```

The dashboard lists every project, shows whether it has been trained, and lets
you start any of them. Each launched project runs as a **separate** Streamlit
server on its own port (8601 and upward); the dashboard tracks the child
processes so they can be stopped again, and records them in
`.streamlit_servers.json` so a restarted dashboard can clean up orphans.

### A single project

Every project is fully self-contained. From a project directory:

```bash
cd 03-handwritten-digit-recognizer
python train.py          # creates models/*.pt  (skip if already trained)
streamlit run app.py     # launches just this project
```

Project 10 has no training step — its Whisper weights download on first use.

## Training the models

Train everything from the repository root:

```bash
./scripts/train_all.sh
```

Or a subset by project number:

```bash
./scripts/train_all.sh 01 07 08
```

Each `train.py` logs a comparison table (all candidate models on the same split),
selects a winner by a stated criterion, and writes both the model artefact and a
`training_metrics.json` into the project's `models/` directory. Every project
supports useful flags — for example:

```bash
# 03 - reproduce a longer run, or force CPU
python train.py --epochs 5 --device cpu

# 04 - more users in the offline evaluation, content model only
python train.py --eval-users 400 --skip-collaborative

# 08 - transfer learning instead of the from-scratch CNN
python train.py --architecture mobilenetv2
```

Some projects download their dataset on the first training run; each is
documented in the [datasets](#datasets-and-licensing) table below.

## Testing

```bash
./scripts/run_tests.sh                 # run everything
./scripts/run_tests.sh -m "not network"  # skip tests that need the internet
./scripts/run_tests.sh -k stock        # extra pytest args are forwarded
```

The suite is organised as one test module per project
(`tests/test_stock_predictor.py`, `tests/test_sentiment.py`, …). Each test module
is executed in its **own pytest process** (see
[challenge #1](#1-the-src-package-name-collision) for why), and each one calls
`tests/project_env.use_project()` before importing `src` so its imports resolve
deterministically to the right project.

Tests are marked `slow` (training or a download) and `network` (needs internet),
so a quick, fully offline run is one flag away.

## Datasets and licensing

No dataset is committed to Git. Each project downloads (or generates) its data on
first use and caches it under its own `data/` directory, together with a
`.download_manifest.json` recording the size and SHA-256 digest of every file.

| # | Dataset | Source | Approx. size | Notes |
| --- | --- | --- | --- | --- |
| 01 | `stockdata.csv`, `finance-charts-apple.csv` | Plotly datasets mirror | ~0.2 MB | Real market data (2007–2017 snapshots). No API key. |
| 02 | Twitter US Airline Sentiment | public `Tweets.csv` mirror | ~3.4 MB | Real hand-labelled tweets; three classes. |
| 03 | MNIST | `ossci-datasets` S3 mirror | ~12 MB | Official mirror used by PyTorch. |
| 04 | MovieLens `ml-latest-small` | GroupLens | ~1 MB | No registration required. |
| 05 | IBM Telco Customer Churn | IBM public ICP4D sample repo | ~0.9 MB | No login; no Kaggle click-through. |
| 06 | Customer-service intents | hand-written, in-repo | tiny | No download; fully reviewable. |
| 07 | UCI SMS Spam Collection | UCI ML Repository | ~0.2 MB | CC BY 4.0. |
| 08 | Fruits-360 | Hugging Face parquet mirror | ~98 MB | One parquet file, 22,688 images. |
| 09 | Open-Meteo historical archive | Open-Meteo API | small | Key-free daily observations. |
| 10 | Whisper weights | Hugging Face Hub | 75–1500 MB | Downloaded on first transcription; runs offline afterwards. |

Substitutions are always documented in the code, the interface and this README
(for example, project 07 is trained on SMS because that is the one openly
redistributable *labelled* spam corpus, and project 02 uses a public tweet
dataset rather than scraping Twitter/X).

## Engineering challenges and how they were solved

These are the real problems encountered while building the suite, and the fixes
that are in the code today. They are listed roughly in order of impact.

### 1. The `src` package name collision

**Problem.** Every project ships a package literally named `src` (that layout is
required). Python caches modules by name in `sys.modules`, so importing two
projects' `src` packages into one interpreter is impossible — the second import
silently returns the first project's modules. It was reproduced directly during
development:

```
ImportError: cannot import name 'CLASS_LABELS' from 'src.data'
(.../01-stock-price-predictor/src/data.py)
```

The same collision also breaks `pickle`: a class pickled as
`src.model.SentimentModel` cannot be re-imported once `src` points at a different
project.

**Fix.** Two complementary solutions:

- **At runtime**, `shared/launcher.py` starts each project's Streamlit app in its
  own process with its own working directory, so only one `src` is ever imported
  per process. The dashboard tracks ports, PIDs and health checks so it can start
  and stop app servers cleanly.
- **In the tests**, `scripts/run_tests.sh` runs one pytest process per test
  module, and `tests/project_env.use_project()` evicts any cached `src.*` module
  and puts exactly one project directory at the front of `sys.path`. This is
  explicit and deterministic — no import-time magic.

### 2. Leakage in time-series and text features

**Problem.** Random shuffling before a chronological problem, or fitting a
TF-IDF vectoriser on the full dataset, leaks future/held-out information into
training and inflates the reported score.

**Fix.** `shared/ml.py` provides a chronological split
(`chronological_split`) and *deliberately does not offer random shuffling* for
time series, making accidental leakage harder. Projects 02/05/07 wrap the
vectoriser and the classifier in a single scikit-learn `Pipeline`, so the
vocabulary and IDF weights are fitted on the training split only. Project 01 has
a dedicated test (`test_features_are_free_of_future_leakage`) that changes a
future price and asserts no past feature row changes.

### 3. Live market data became unavailable

**Problem.** The original plan was to fetch live prices with `yfinance`, but the
Yahoo endpoint returned **HTTP 429** for every request, and the usual Stooq
fallback now serves a JavaScript anti-bot challenge instead of CSV.

**Fix.** Project 01 switched to two small, stable, publicly hosted CSV datasets
(Plotly's mirrors of the Quandl `stockdata.csv` and `finance-charts-apple.csv`).
They need no API key and download in under a second, so training stays fully
reproducible. The project now makes no claim about *current* prices, because the
data is a historical snapshot.

### 4. A fruit model that reported 100% — and why that was a red flag

**Problem.** A first version of project 08 scored **100% test accuracy**, which is
not credible on this task. Investigation showed the mirror ships many images more
than once: a check of 150 random images found that *every one* had a byte-identical
twin elsewhere, so a random split put copies of test images into training.

**Fix.** Exact deduplication — images are hashed by decoded pixel bytes and only
the first occurrence is kept. A difference-hash/union-find approach was tried
first and rejected: its distance distribution had no gap separating "copy" from
"different photo" and chained 499 clusters together (one holding 2,318 images).
Exact hashing removes genuine copies and never merges two different photographs.
*Near*-duplicates (different photos of the same physical fruit) cannot be
separated reliably, so they are left in place and stated as a limitation.

### 5. The "fruit" dataset that was too small to learn from

**Problem.** The obvious Fruits-30 collection has `apples` with only 28 images and
`bananas` with just 11 — a CNN trained on that memorises eleven photographs.

**Fix.** Project 08 uses **Fruits-360** (22,688 images delivered as a single ~98 MB
parquet file), and maps its 113 fine-grained *variety* labels up to seven
*fruits* by prefix (`Apple Braeburn`, `Apple Granny Smith`, … → `Apple`). The
grouping keeps 5,150 images across seven classes — a harder, more honest task
than the three suggested.

### 6. scikit-learn version quirks that broke "standard" tutorials

Three separate issues, all fixed in the shared layer:

- The `liblinear` solver raises `ValueError` for three-class problems in current
  scikit-learn; project 02 uses `lbfgs`, which handles the multinomial case
  natively.
- `precision_score`/`recall_score` with the default integer `pos_label=1` raise
  `ValueError: pos_label=1 is not a valid label` for string labels.
  `shared/metrics.py` accepts an explicit `pos_label`.
- `confusion_matrix(..., labels=["Apple", "Banana"])` fails when the model works
  on integer indices. The same module accepts a separate `label_names` argument
  so display names and internal label values can differ.

### 7. Fragile dataset downloads corrupting training

**Problem.** An interrupted download leaves a half-written CSV. A truncated file
is one of the most confusing failure modes — training "works" and produces
nonsense rather than an error.

**Fix.** `shared/datasets.py` downloads to a temporary file and atomically renames
it into place, verifies a SHA-256 digest recorded in a manifest, retries across
mirrors with backoff, and rejects archive members that would escape the
destination directory (zip-slip protection).

### 8. Cross-platform model persistence and devices

**Problem.** `torchvision` returns tensors that cannot be archived as portably as
NumPy arrays, and re-downloading MNIST or re-decoding 22k JPEGs on every page load
is unacceptable.

**Fix.** Projects decode once and cache to a single `.npz`
(`data/mnist.npz`, `fruits360_cache.npz`); the app and the test suite then load in
milliseconds offline. `shared/ml.py` picks the best available device
(CUDA → Apple Metal → CPU) so nothing ever *requires* a GPU.

### 9. Streamlit and Matplotlib API friction

**Problem.** Streamlit rejects an explicit `height=None` on `st.dataframe`, and
un-closed Matplotlib figures leak memory across reruns.

**Fix.** `shared/ui.py` omits the `height` argument entirely when none is
requested, and closes Matplotlib figures right after rendering.

### 10. Confidently wrong chatbots

**Problem.** A bag-of-words intent classifier maps every off-topic message to
whichever support intent shares the most words, so a chatbot answers questions it
should refuse.

**Fix.** Project 06 adds an explicit `unrecognized` intent trained on genuine
chit-chat, plus a **confidence threshold** and a **margin** between the top two
classes. Below either bound, the bot declines to answer instead of guessing.

### 11. Optional dependencies that must never break a fresh clone

**Problem.** NLTK's stop-word corpus is not always present offline; an optional
`python-dotenv` should not be mandatory.

**Fix.** `shared/text.py` prefers `nltk.corpus.stopwords` when it is installed
*and* downloaded, and otherwise falls back to a bundled English stop-word list —
so a fresh clone never fails because a corpus is missing. `shared/config.py`
reads secrets from real environment variables and only uses `.env` when
`python-dotenv` happens to be installed; every variable has a working default.

## Known limitations

The suite is deliberately honest about what it does **not** do:

- Project 01 models are frequently *not* better than the naive persistence
  baseline; the interface says so on the ticker where that holds.
- Project 07 is trained and evaluated on **SMS**, not email. The reported
  accuracy is an SMS number and should not be quoted as an email one.
- Project 08's reported accuracy is optimistic because *near*-duplicates cannot
  be removed reliably (see challenge #4).
- Forecasting projects (01, 09) are educational and make no claim about live or
  future prices/weather.

## License

Released under the **MIT License** — see [LICENSE](LICENSE). Datasets retain the
licenses of their original sources, as noted in the datasets table above.