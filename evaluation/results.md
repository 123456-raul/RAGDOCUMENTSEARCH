# Retrieval evaluation

20 questions, top 4 chunks per question, 20 candidates before merging and reranking.

- **Hit rate**: share of questions where a returned chunk contains the answer.
- **MRR**: mean of 1 / (position of the first such chunk); 1.0 means always first.
- **Compared**: share of chunk embeddings the query was compared with.
- **Search**: average time per question on the machine that ran this.

| Chunks (size / overlap, count) | Index | Pipeline | Hit rate | MRR | Compared | Search |
|---|---|---|---|---|---|---|
| 500 / 50, 328 | flat | embeddings only | 65% | 0.62 | 100% | 9 ms |
| 500 / 50, 328 | flat | + keywords (BM25) | 70% | 0.63 | 100% | 9 ms |
| 500 / 50, 328 | flat | + reranker | 75% | 0.69 | 100% | 429 ms |
| 500 / 50, 328 | flat | + keywords + reranker | 75% | 0.69 | 100% | 455 ms |
| 500 / 50, 328 | tree | embeddings only | 45% | 0.45 | 19% | 9 ms |
| 500 / 50, 328 | tree | + keywords (BM25) | 60% | 0.46 | 19% | 10 ms |
| 500 / 50, 328 | tree | + reranker | 65% | 0.55 | 19% | 383 ms |
| 500 / 50, 328 | tree | + keywords + reranker | 80% | 0.74 | 19% | 453 ms |
| 1000 / 150, 159 | flat | embeddings only | 75% | 0.59 | 100% | 10 ms |
| 1000 / 150, 159 | flat | + keywords (BM25) | 80% | 0.74 | 100% | 9 ms |
| 1000 / 150, 159 | flat | + reranker | 85% | 0.85 | 100% | 898 ms |
| 1000 / 150, 159 | flat | + keywords + reranker | 90% | 0.90 | 100% | 907 ms |
| 1000 / 150, 159 | tree | embeddings only | 75% | 0.59 | 33% | 10 ms |
| 1000 / 150, 159 | tree | + keywords (BM25) | 85% | 0.72 | 33% | 10 ms |
| 1000 / 150, 159 | tree | + reranker | 85% | 0.85 | 33% | 686 ms |
| 1000 / 150, 159 | tree | + keywords + reranker | 90% | 0.90 | 33% | 892 ms |

## Questions the full pipeline (keywords + reranker) missed

- 500 / 50, 328 chunks, flat: How long did it take to train the base Transformer model?
- 500 / 50, 328 chunks, flat: What GPUs was the Transformer trained on?
- 500 / 50, 328 chunks, flat: What is the size of the inner layer of the feed-forward network?
- 500 / 50, 328 chunks, flat: How does the model know the order of tokens without recurrence or convolution?
- 500 / 50, 328 chunks, flat: Which dataset was used for the English constituency parsing experiments?
- 500 / 50, 328 chunks, tree: What GPUs was the Transformer trained on?
- 500 / 50, 328 chunks, tree: What is the size of the inner layer of the feed-forward network?
- 500 / 50, 328 chunks, tree: How does the model know the order of tokens without recurrence or convolution?
- 500 / 50, 328 chunks, tree: Which dataset was used for the English constituency parsing experiments?
- 1000 / 150, 159 chunks, flat: What GPUs was the Transformer trained on?
- 1000 / 150, 159 chunks, flat: How does the model know the order of tokens without recurrence or convolution?
- 1000 / 150, 159 chunks, tree: What GPUs was the Transformer trained on?
- 1000 / 150, 159 chunks, tree: How does the model know the order of tokens without recurrence or convolution?
