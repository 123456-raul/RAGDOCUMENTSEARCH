# Retrieval evaluation

20 questions, top 4 chunks per question, 20 candidates before merging and reranking.

- **Hit rate**: share of questions where a returned chunk contains the answer.
- **MRR**: mean of 1 / (position of the first such chunk); 1.0 means always first.
- **Compared**: share of chunk embeddings the query was compared with.
- **Search**: average time per question on the machine that ran this.

| Chunks (size / overlap, count) | Index | Pipeline | Hit rate | MRR | Compared | Search |
|---|---|---|---|---|---|---|
| 500 / 50, 328 | flat | embeddings only | 75% | 0.69 | 100% | 9 ms |
| 500 / 50, 328 | flat | + keywords (BM25) | 75% | 0.72 | 100% | 9 ms |
| 500 / 50, 328 | flat | + reranker | 85% | 0.79 | 100% | 434 ms |
| 500 / 50, 328 | flat | + keywords + reranker | 85% | 0.79 | 100% | 453 ms |
| 500 / 50, 328 | tree | embeddings only | 55% | 0.53 | 19% | 8 ms |
| 500 / 50, 328 | tree | + keywords (BM25) | 65% | 0.54 | 19% | 9 ms |
| 500 / 50, 328 | tree | + reranker | 70% | 0.63 | 19% | 395 ms |
| 500 / 50, 328 | tree | + keywords + reranker | 90% | 0.84 | 19% | 458 ms |
| 1000 / 150, 159 | flat | embeddings only | 85% | 0.67 | 100% | 9 ms |
| 1000 / 150, 159 | flat | + keywords (BM25) | 90% | 0.84 | 100% | 9 ms |
| 1000 / 150, 159 | flat | + reranker | 95% | 0.95 | 100% | 933 ms |
| 1000 / 150, 159 | flat | + keywords + reranker | 100% | 1.00 | 100% | 947 ms |
| 1000 / 150, 159 | tree | embeddings only | 85% | 0.67 | 33% | 9 ms |
| 1000 / 150, 159 | tree | + keywords (BM25) | 90% | 0.80 | 33% | 9 ms |
| 1000 / 150, 159 | tree | + reranker | 95% | 0.95 | 33% | 684 ms |
| 1000 / 150, 159 | tree | + keywords + reranker | 100% | 1.00 | 33% | 928 ms |

## Questions the full pipeline (keywords + reranker) missed

- 500 / 50, 328 chunks, flat: How long did it take to train the base Transformer model?
- 500 / 50, 328 chunks, flat: What is the size of the inner layer of the feed-forward network?
- 500 / 50, 328 chunks, flat: Which dataset was used for the English constituency parsing experiments?
- 500 / 50, 328 chunks, tree: What is the size of the inner layer of the feed-forward network?
- 500 / 50, 328 chunks, tree: Which dataset was used for the English constituency parsing experiments?
