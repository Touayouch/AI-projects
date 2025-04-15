# Évaluation du retrieval — chunking `semantic`

44 questions annotées (`eval/retrieval_questions.jsonl`), pertinence au niveau fiche.

## Global

| Méthode | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | Latence |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 0.68 | 0.80 | 0.86 | 0.89 | 0.753 | 147 ms |
| sparse | 0.55 | 0.64 | 0.68 | 0.73 | 0.598 | 10 ms |
| hybrid | 0.75 | 0.80 | 0.84 | 0.86 | 0.788 | 102 ms |

## Par type de question

| Méthode | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | Latence |
| --- | --- | --- | --- | --- | --- | --- |
| dense (lexical) | 0.90 | 0.95 | 1.00 | 1.00 | 0.938 | 76 ms |
| sparse (lexical) | 0.85 | 1.00 | 1.00 | 1.00 | 0.908 | 10 ms |
| hybrid (lexical) | 0.95 | 1.00 | 1.00 | 1.00 | 0.975 | 88 ms |
| dense (paraphrase) | 0.50 | 0.67 | 0.75 | 0.79 | 0.600 | 85 ms |
| sparse (paraphrase) | 0.29 | 0.33 | 0.42 | 0.50 | 0.339 | 13 ms |
| hybrid (paraphrase) | 0.58 | 0.62 | 0.71 | 0.75 | 0.632 | 103 ms |
