# Sabche detection: Gemini and Claude zero-shot vs a fine-tuned mmBERT

Sabche (ས་བཅད) is the outline heading a Tibetan commentary gives to each section it is about to explain, for example `གཉིས་པ་༼ཚུལ་ཁྲིམས་ཕར་ཕྱིན་གྱི་རབ་དབྱེ་༽ནི`. This repo scores three ways of finding these headings in whole books on the same 29 held-out test books: Gemini 3.1 Flash Lite and Claude Sonnet 5, each with the same prompt and no training, and a fine-tuned mmBERT.

## Results

F1 on the test split (4,079 gold headings). A prediction counts when its character overlap (IoU) with a gold span is at least 0.5. Precision and recall are in brackets.

| | Gemini 3.1 Flash Lite | Claude Sonnet 5 | mmBERT (sabche v1) |
|---|---|---|---|
| All 29 books | 0.668 (P 0.600, R 0.754) | 0.681 (P 0.613, R 0.766) | **0.962** (P 0.954, R 0.970) |
| Old-batch books (11) | 0.547 (P 0.458, R 0.678) | 0.643 (P 0.533, R 0.809) | **0.958** (P 0.940, R 0.977) |
| New-batch books (18) | 0.762 (P 0.724, R 0.804) | 0.712 (P 0.688, R 0.737) | **0.964** (P 0.964, R 0.965) |
| Median book | 0.623 | 0.684 | **0.979** |
| Spans predicted (gold: 4,079) | 5,129 | 5,096 | 4,144 |

The trained model is far ahead of both prompted models. Claude Sonnet 5 and Gemini are close overall (0.681 against 0.668, one run each, so treat them as roughly equal), but they differ by batch: Claude is better on the old-batch books (0.643 against 0.547) and worse on the new-batch ones (0.712 against 0.762). Its worst new-batch books are I3F4A91F5 (0.591 against Gemini's 0.875) and IFA88A536 (0.613 against 0.891); why was not looked into.

On Gemini: Gemini finds most of the headings (recall 0.754) but predicts about 1,000 more spans than there are gold headings. Its precision (0.600 against 0.954) is the bigger gap, and its recall is lower too (0.754 against 0.970). The worst cases are P000083 (old-batch; 344 extra predictions against 1 for mmBERT) and IC05A6BE0 (new-batch; 159 extra against 8 gold spans). The gold does not mark every heading in a book, and a model trained on it can learn which ones are marked, while the prompted model cannot. Many of the extra predictions are therefore probably real headings that were never annotated, but they were not checked one by one.

mmBERT's own test evaluation, which scores per window on tokens, gives F1 0.965 (`results/mmbert-sabche-v1/test_eval_window_level.json`). The book-level character numbers above are the ones comparable to Gemini.

## Check the numbers

Predictions are stored as character offsets, so scoring needs no book texts and no API keys.

```
python scripts/score_spans.py --split test --per-book \
  --model gemini-3.1-flash-lite=results/gemini-3.1-flash-lite/test/spans \
  --model claude-sonnet-5=results/claude-sonnet-5/test/spans \
  --model mmbert-sabche-v1=results/mmbert-sabche-v1/test/spans
```

## Run it again

The book texts come from OpenPecha and are not in this repo. Fetch them first:

```
python scripts/fetch_texts.py --ids-file data/ids.txt --raw-dir data/raw_opf \
    --manifest data/raw_opf/_manifest.csv
```

Gemini (set `GEMINI_API_KEY`):

```
python scripts/zeroshot_run.py --texts-dir data/raw_opf --books I9B6A4525 \
    --provider gemini --model gemini-3.1-flash-lite --out runs/gemini
```

Each book is cut into 16,000-character windows that overlap by 2,000. The model returns the first and last ~20 characters of each heading and `src/gemini_locate.py` finds them in the window. Replies are cached per window, so a repeat run costs nothing for finished windows, and `--stop-after 1` pauses after each book. The test set is 540 windows. The Gemini cost was not tracked. The prompt is `prompts/gemini_sabche_anchors_v1.md`.

Claude Sonnet 5 uses the same windows, prompt and locator. It was run through the Message Batches API (half price, asynchronous) with `scripts/claude_batch.py`, in three batches of about 180 windows each (`plan`, `create --yes`, `poll`, `fetch`), after a 2-window canary batch; that is all 540 test windows. Settings: adaptive thinking, effort low, JSON-schema output, `max_tokens` 64,000. The third batch (179 windows) cost about $5.34 at batch prices; the cost of the first two was not recorded. `scripts/zeroshot_run.py --provider anthropic --model claude-sonnet-5` runs the same thing one window at a time. Fetched replies are cached per window, and `--locate-only` on `zeroshot_run.py` turns them into spans without any API call.

The mmBERT model is `Yontenn/mmbert-sabche-v1` on Hugging Face. If the page gives a 404 the repo is still private and needs access from its owner. It was trained on the v1 split; all 29 test books here were in that split's test set, so it never trained on them. Its reported predictions come from a GPU run and were converted to character offsets with `scripts/mmbert_dump_to_chars.py`.

## What is in the repo

- `data/`: gold spans as offsets (`sabche_gold_v2.csv`, 42,292 spans over 323 books), each book's split and batch (`books_manifest.csv`, 261 train, 33 validation, 29 test), and the 10 books left out for being under-annotated (`excluded_books.csv`).
- `results/`: predicted spans for all three models, plus the per-book scores of Gemini and Claude and mmBERT's own evaluation file.
- `pipeline/`: the scripts that built the gold, the split and the tokenized dataset, with a README on each step. Reference only, since they need the working repo's tsawa code.
- `prompts/` and `src/`: the Gemini prompt, the window cutter and the anchor locator, which are copied unchanged from the quotation benchmark.

Book texts, tokenized data and raw model replies (including Claude's) are not included. The texts belong to the OpenPecha repositories, and the fetch script notes that data-rights questions there are still open, so check their terms before sharing any text.

Some kept books are only partly annotated, and bare `Nth-པ་ནི།` headings are annotated only about 29% of the time, so some of Gemini's false positives may be gaps in the gold.
