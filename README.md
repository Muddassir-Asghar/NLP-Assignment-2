# NLP Assignment 2

This project contains one notebook that implements all assignment parts:

- i23-2577_Assignment2_DS-A.ipynb

The instructions below explain how to reproduce each part from scratch and where outputs are saved.

## 1) Environment Setup

### Prerequisites

- Python 3.10+
- Jupyter Notebook or VS Code with Jupyter extension

### Install Dependencies

Use your preferred environment manager, then install:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch openTSNE
```

If CUDA is unavailable or incompatible, training will run on CPU.

## 2) Project Structure

Important folders used by the notebook:

- data/: cleaned corpus, token artifacts, CoNLL files, Part 3 arrays/splits
- embeddings/: TF-IDF matrix and word embeddings artifacts
- models/: saved BiLSTM and Transformer checkpoints
- init_data/: intermediate annotation files used in sequence labeling prep

## 3) How to Reproduce Everything

1. Open i23-2577_Assignment2_DS-A.ipynb.
2. Restart kernel.
3. Run all cells from top to bottom in order.

This is the most reliable path because later parts reuse variables and files produced by earlier parts.

## 4) Part-by-Part Reproduction

## Part 1: Embeddings and Analysis

Goal: Build and evaluate multiple embedding representations (TF-IDF, PPMI, Word2Vec variants) and compare behavior.

### Steps

1. Run the Part 1 preprocessing and token loading cells.
2. Run TF-IDF generation cells.
3. Run PPMI + dimensionality reduction/visualization cells.
4. Run Skip-gram Word2Vec training/evaluation cells.
5. Run nearest-neighbor and analogy comparison cells.

### Main outputs

- embeddings/tfidf_matrix.npy
- embedding-related reports/plots generated in notebook output
- token/json artifacts under data/ (if regenerated)

## Part 2: Sequence Labeling (POS + NER with BiLSTM)

Goal: Prepare labeled sequence data, train BiLSTM models (POS and NER-CRF), and evaluate.

### Steps

1. Run dataset-preparation cells that create POS/NER annotations and CoNLL splits.
2. Run BiLSTM setup cells (paths, loaders, vocab, embedding init).
3. Run POS training cells (frozen and fine-tuned embeddings).
4. Run NER training cells (CRF decoder; frozen and fine-tuned).
5. Run evaluation cells for POS and NER.

### Main outputs

- data/pos_train.conll, data/pos_val.conll, data/pos_test.conll
- data/ner_train.conll, data/ner_val.conll, data/ner_test.conll
- models/bilstm_pos.pt
- models/bilstm_ner.pt
- embeddings/word2idx.json

### Note on NER zeros

If entity-level precision/recall/F1 become all zeros, this usually means predictions collapsed mostly to O due to severe class imbalance and sparse entity spans.

## Part 3: Topic Classification with Transformer Encoder

Goal: Build a custom Transformer encoder from scratch (no built-in Transformer classes), train it for 5-class topic prediction, and evaluate with attention visualization.

### Steps

1. Run dataset prep cells for topic classification:
   - keyword-based category assignment
   - token-id sequences padded/truncated to length 256
   - 70/15/15 split
2. Run Transformer module cells in order:
   - scaled dot-product attention
   - multi-head self-attention
   - feed-forward network
   - sinusoidal positional encoding
   - encoder block/stack
   - CLS classification head
   - full model
3. Run training cell (AdamW, warmup + cosine schedule, 20 epochs).
4. Run result cells:
   - test accuracy and macro-F1
   - 5x5 confusion matrix
   - attention heatmaps for correctly classified samples
5. Run the BiLSTM vs Transformer comparison markdown section.

### Main outputs

- data/p3_article_categories.json
- data/p3_token_ids_256.npy
- data/p3_true_lengths.npy
- data/p3_category_ids.npy
- data/p3_stratified_split.json
- models/transformer_topic_classifier.pt

## 5) Quick Re-run by Part

If you only want to re-run one part:

- Part 1 only: restart kernel and run through the last Part 1 cell.
- Part 2 only: run Part 2 data prep first, then BiLSTM training/evaluation cells.
- Part 3 only: run Part 3 data prep, then all Transformer cells and results cells.

If a section depends on variables from earlier sections, re-run required setup cells first.

## 6) Reproducibility Tips

- Keep the random seed cells unchanged.
- Do not shuffle cell order.
- If outputs look inconsistent, restart kernel and run all cells again.
- Verify required input files exist in data/, embeddings/, and init_data/ before training.
