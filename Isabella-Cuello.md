# Contribution summary — Isabella Cuello

ISM 6564 · Assignment 7 · Owner of Task 1 (the index, cut two ways) and Task 2 (the search)

The code for both tasks was written with AI help (Claude). I pasted, ran and tested it, made the design choices from measured results, and wrote the "Your answer" cells myself. Details of AI use are in `ai-usage/Isabella-Cuello.md`.

---

## Task 1 — The index, cut two ways

**What I built**
- `chunk(doc)`: the section cut. One passage per `##` section, with the title line on top. Title-only pieces are skipped and the labels are copied from the article.
- `chunk_semantic(doc, threshold)`: the semantic cut. Embeds sentences inside the function and compares each sentence with the next real sentence. It cuts when similarity is below the threshold and the passage has at least `MIN_CHARS`. A small leftover joins the previous passage. Headings are never compared.
- Settings: `SEMANTIC_THRESHOLD = 0.47`, `MIN_CHARS = 250`, `CUT = "semantic"`.

**Decisions and why**
- Threshold candidates came from percentiles of 424 neighbour-sentence similarities: 25th = 0.21, 50th = 0.33, 75th = 0.47.
- I tested all three against the section cut and chose 0.47:

| Cut | Passages | Answers whole | Recall@5 | Fact in window | Tokens in window | Too long for encoder |
| --- | --- | --- | --- | --- | --- | --- |
| Section | 142 | 27/27 | 0.89 | 0.85 | 412 | 0 |
| Semantic 0.21 | 79 | 27/27 | 0.87 | 0.78 | 716 | 2 |
| Semantic 0.33 | 102 | 27/27 | 0.87 | 0.89 | 532 | 1 |
| **Semantic 0.47 (chosen)** | 119 | 27/27 | 0.94 | 0.96 | 458 | 0 |

- Boundary example (KB-001):
  - The semantic cut joined "Fees" and "Surcharges" into one passage, which G22 needs (Expedited fee + Alaska surcharge).
  - It also split "Services and delivery windows" between the order cut-off time and the delivery-days list.

**Results**
- `chunk`: 4/4 checks PASS. `chunk_semantic`: 9/9 checks PASS.
- The index is built from the semantic cut: 119 passages.

**Written:** the Task 1 "Your answer" cell, drafted and revised by me (AI critique only).

---

## Task 2 — The search

**What I built**
- `rrf(*rankings, c=60)`: merges ranked lists. Each passage gets 1/(c + rank) per list, summed, best first.
- `search(question, k, allowed=None)`: takes the top 20 from `meaning_search` and the top 20 from `keyword_search`, fuses them with `rrf`, re-ranks them with the cross-encoder, and returns the best k. It passes `allowed` to both tools.
- Settings: `SHORTLIST = 20`, `USE_RERANK = True`, `K = 7`.

**Decisions and why**
- **Re-ranking on.** At k = 5, fact in window went from 0.85 to 0.93 and the semantic questions from 0.50 to 0.75. Misses dropped from 4 to 2.
- **Shortlist of 20 per tool.** A middle ground between coverage and speed. I did not measure other sizes.
- **K = 7.** The smallest k with fact in window = 1.00. k = 10 added nothing but cost more tokens.

| k | Recall | Fact in window | Tokens in window |
| --- | --- | --- | --- |
| 1 | 0.67 | 0.59 | |
| 3 | 0.87 | 0.89 | |
| 5 | 0.94 | 0.93 | 485 |
| **7 (chosen)** | 0.98 | 1.00 | 662 |
| 10 | 0.98 | 1.00 | 930 |

**Results (K = 7)**

| Search | Recall@k | Fact in window | Lexical | Semantic | Multi_doc | Cross_section | Stale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meaning only | 0.94 | 0.96 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| Keyword only | 0.80 | 0.78 | 1.00 | 0.38 | 0.80 | 1.00 | 1.00 |
| **My search** | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- All 4 Task 2 checks PASS, and no question's fact is missing from the window at K = 7.
- Near miss at k = 5, G09 (the parcel shows as delivered but isn't there): the KB-003 passage with "24 hours" ranked 6th, behind KB-002 tracking passages. K = 7 brings it into the window.

**Written:** the Task 2 "Your answer" cell, drafted and revised by me over 3 rounds (AI critique only).
