# Preprocessing ablation (sanity probe: TF-IDF 1-2 grams + LinearSVC, validation split)

Raw data already contains 7 texts attached to more than one intent (label noise).

| Variant | Vocab | Empty rows | Conflicting texts | Val accuracy | Val macro-F1 |
|---|---:|---:|---:|---:|---:|
| lowercase_only | 6,110 | 0 | 8 | 0.8624 | 0.8473 |
| legacy_v1 | 5,420 | 24 | 134 | 0.8270 | 0.7805 |
| v2_function_words | 5,244 | 0 | 84 | 0.8398 | 0.8299 |
| v2_final | 5,257 | 0 | 40 | 0.8453 | 0.8325 |
| v2_keep_all_words | 5,314 | 0 | 12 | 0.8605 | 0.8339 |
