# How the Sabche data was made

These scripts turn the raw OpenPecha `Sabche.yml` layers into the gold spans, the split and the tokenized dataset used in this repo. They are here for reference. They were run from the working repo, where they import `build_tsawa_dataset.py` from `src/tsawa/`, and that file is not in this repo, so they will not run as they stand. The outputs are in `../data/`.

## The four steps

| Step | Command | Reads | Writes |
|---|---|---|---|
| 0. Audit (optional) | `python scratch/sabche/scripts/audit_sabche.py` | raw `Sabche.yml` layers and book text | numbers only, changes nothing |
| 1. Clean the spans | `python src/sabche/clean_sabche_spans.py` | raw `Sabche.yml` in `data/raw_opf/` | `sabche_spans_clean.csv`, `sabche_book_verdicts.csv` |
| 2. Split the books | `python src/sabche/prepare_sabche_split.py --val-frac 0.085 --test-frac 0.085 --version v2` | cleaned spans, `split_v3_frozen.csv` (the tsawa split) | `sabche_split_v2_frozen.csv`, `sabche_excluded_books.csv` |
| 3. Tokenize | `python src/sabche/build_sabche_dataset.py --split-file data/processed/sabche/sabche_split_v2_frozen.csv --out-dir data/processed/sabche/sabche_dataset_v2 --stats-json data/processed/sabche/sabche_dataset_v2_stats.json` | cleaned spans, the split, book text | the Arrow dataset and a stats file |

All outputs go to `data/processed/sabche/` in the working repo. Run the commands from that repo's root.

## What each step does

**Clean.** Some books, mostly from the old batch, have offsets shifted by a constant amount that changes part-way through (for example P000063: +116, then +229, then +291). The script finds anchor headings (ordinal headings such as `དང་པོ … ནི` whose length matches the span) and picks one offset per span with a Viterbi pass that pays a cost for every change of offset. It accepts the shift only if the share of spans that look like headings rises enough. It then snaps cuts that fall inside a syllable to the nearest boundary, up to 3 characters. Ends stop before the boundary character, because sabche spans exclude the trailing shad, unlike tsawa. Anything further off is left alone. 383 of the 43,223 spans are marked dropped: 227 are realigned spans that do not verify as headings, and 156 belong to books excluded because their offsets cannot be recovered. Adjacent spans are never merged, since each is a separate outline heading. `Sabche.yml` is never changed.

Each book gets a verdict: 298 keep, 30 realigned, 5 exclude (`../data/sabche_book_verdicts.csv`).

**Split.** Books keep the split they have in the tsawa split (`split_v3_frozen.csv`). Books that share text stay together: two books are linked when at least 10% of one book's long-span text appears verbatim in the other over at least 2 spans, or when 10 or more spans of 40+ characters are shared (a single shared boilerplate title does not count). A group whose fixed members disagree goes to test, then validation, then train. The remaining groups fill the split by window count. Ten books are excluded: 5 whose outline is annotated but whose body headings are not, 3 with unrecoverable offsets and 2 with no heading-like spans (`../data/excluded_books.csv`). The result is 261 train, 33 validation and 29 test books.

**Tokenize.** It reuses the tsawa builder's labelling and windowing (`label_tokens`, `sliding_windows`, `pack_window`) with a sabche label map (`O`, `B-SABCHE`, `I-SABCHE`), 8,192-token windows with a 5,120 stride, and the `jhu-clsp/mmBERT-base` tokenizer. Counts are in `../data/sabche_dataset_v2_stats.json`.

Training and evaluation ran on a separate GPU machine, not from this repo. The training script is the tsawa one (`train_tsawa.py`), which still hard-codes the `TSAWA` label names, so the sabche runs used it with the labels renamed to `B-SABCHE` and `I-SABCHE` in the Hugging Face dataset, according to the notes from those runs.

## Other scripts

- `show_sabche_check.py` builds an HTML page for eyeballing the cleaned spans (8 books, headings highlighted). It contains book text, so the page itself is not in the repo.
- `pred_ordinal_check.py` checks whether predicted sabche spans start on an ordinal as often as gold spans do. It is self-contained and was run on the GPU machine, using only the model and the dataset from Hugging Face.
