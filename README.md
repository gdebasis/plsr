# PLSR (Probabilistic Learned Sparse Retrieval)

This code is distributed in the hope that it'll be useful for IR practitioners and students who want to get started with retrieving documents from a collection and measure effectiveness with standard evaluation metrics.


This repository implements a **global probabilistic term weighting model** inspired by the classical Robertson–Spärck Jones (RSJ) framework. The goal is to estimate term importance using supervised relevance signals derived from a set of training queries and relevance judgments.

Unlike traditional per-query estimation, this implementation aggregates statistics **globally across queries**, resulting in a single collection-level weight for each term.

---

## Theoretical Background

Let:

* ( \mathcal{Q} ): set of training queries
* ( \mathcal{D} ): document collection
* ( R_q \subseteq \mathcal{D} ): set of relevant documents for query ( q )
* ( V ): vocabulary of query terms

We define the importance of a term ( w \in V ) using:

[
\beta_w = \log \frac{P(w \mid R)}{P(w \mid \neg R)}
]

where:

* ( R ): set of relevant documents (aggregated across queries)
* ( \neg R ): set of non-relevant documents

### Term Statistics

* ( r_w = |{d \in R : w \in d}| ): number of relevant documents containing term ( w )
* ( n_w = |{d \in \mathcal{D} : w \in d}| ): number of documents containing ( w )

### Probability Estimates

[
P(w \mid R) = \frac{r_w}{|R|}, \qquad
P(w \mid \neg R) = \frac{n_w - r_w}{|\mathcal{D}| - |R|}
]

This leads to a log-odds formulation closely related to the **RSJ term weight**.

---

## Implementation Details

### Global Estimation

Instead of computing ( r_w ) per query, this implementation aggregates:

[
R_w = \sum_{q \in \mathcal{Q}} |{d \in R_q : w \in d}|
]

This allows:

* efficient reuse across queries
* stable estimation for sparse terms
* compatibility with large-scale collections (e.g., MS MARCO)

---

### Efficient Computation with Lucene

* **( n_w )** is obtained via:

  ```java
  reader.docFreq(new Term(field, w))
  ```

  (fast, index-backed)

* **( R_w )** is computed by:

  * iterating over relevant documents
  * extracting term vectors
  * aggregating counts

---

## Caching and Reuse

To avoid recomputation, the system persists:

### 1. Relevance Statistics

* File: `rwFile`
* Stores:

  * ( R_w ) (term → count)
  * ( |R| ) (total relevant document instances)

### 2. Term Weights

* File: `betaFile`
* Stores:

  * ( \beta_w ) (term → weight)

---

### Index the MS MARCO passage collection

Download the MS MARCO passage collection file `collection.tsv`, and create a soft link to the file named `data/collection.tsv`.
 
Run the following script to build the index.
```
./index.sh
```

Run the following script to execute retrieval.
```
./retrieve.sh <index dir> <qrels train> <qrels file> <query file>
```

