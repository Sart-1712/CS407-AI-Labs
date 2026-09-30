# Week 8: Answers to Questions 1–14

**Lab:** Bayesian Networks and Autoregressive Language Models ([question PDF](../lab-question/BN_lab.pdf))
**Code and full outputs:** [solution/BN_Lab.ipynb](../solution/BN_Lab.ipynb)

Every number on this page comes from the notebook, run on the lab's six-sentence dataset:

```
the cat sat on the mat        the dog ran to the park
the cat sat on the rug        the cat ran to the park
the dog sat on the mat        the dog sat on the rug
```

Each sentence is padded as `<START> … <END>`. The vocabulary is 10 words, plus `<START>`/`<END>`.

---

## Question 1: Why is the chain-rule decomposition useful for generating text?

$$P(x_1,\dots,x_T) = P(x_1)\prod_{t=2}^{T} P(x_t \mid x_1,\dots,x_{t-1})$$

- **It turns one impossible distribution into many small ones.** A joint distribution over whole sentences has $|V|^T$ outcomes, far too many to write down or sample from directly. Each factor is only a distribution over the vocabulary: "which word comes next, given what has been written so far".
- **Generation becomes sequential sampling.** Sample $x_1$, then $x_2$ given $x_1$, then $x_3$ given $x_1, x_2$, and so on. The product of the factors *is* the joint distribution, so this ancestral sampling draws whole sentences from exactly $P(x_1,\dots,x_T)$. Text is produced left to right, the way it is written.
- **Variable length comes for free.** An `<END>` token lets the model decide at every step whether to stop, so $T$ never has to be known in advance.
- **Each factor can be learned from data** as a next-word prediction problem. The same factors also **score** any sentence (its probability is the product of the factors), so one model supports both generation and evaluation.

The chain rule itself is exact: no assumption has been made yet. The approximation only comes in when the conditioning set is truncated (Question 2).

---

## Question 2: What independence assumption does $X_1 \to X_2 \to \dots \to X_T$ make?

The **first-order Markov assumption**: given the previous word, the current word is conditionally independent of all earlier words.

$$X_t \;\perp\; \{X_1,\dots,X_{t-2}\} \;\mid\; X_{t-1}
\qquad\Longleftrightarrow\qquad
P(X_t \mid X_1,\dots,X_{t-1}) = P(X_t \mid X_{t-1})$$

For example, P(mat | the cat sat on the) = P(mat | the).

- The independence is **conditional**, not marginal. $X_t$ and $X_{t-2}$ are still dependent through the chain $X_{t-2}\to X_{t-1}\to X_t$; observing $X_{t-1}$ blocks that path (d-separation).
- The model also uses **one CPT for every position** $t$ (time-homogeneity / parameter tying): P(w | v) is the same at word 2 as at word 5. That is what lets six short sentences estimate the whole model.

---

## Question 3: The CPT P(next word | current word), and zero-probability transitions

Estimated by counting: $P(w_j \mid w_i) = C(w_i, w_j) / \sum_k C(w_i, w_k)$.

| current word | observations | next-word distribution |
|---|---|---|
| `the` | 12 | cat 3/12 = 0.25, dog 3/12 = 0.25, mat 2/12, rug 2/12, park 2/12 (0.167 each) |
| `cat` | 3 | sat 2/3, ran 1/3 |
| `dog` | 3 | sat 2/3, ran 1/3 |
| `sat` | 4 | on 1 |
| `ran` | 2 | to 1 |

`the` precedes a word 12 times, because it appears twice per sentence. So P(cat | the) = 3/12, not the 3/5 of the lab sheet's illustrative example.

Full first-order CPT (`·` = zero probability):

```
             cat    dog    mat     on   park    ran    rug    sat    the     to  <END>
<START>        ·      ·      ·      ·      ·      ·      ·      ·      1      ·      ·
cat            ·      ·      ·      ·      ·    1/3      ·    2/3      ·      ·      ·
dog            ·      ·      ·      ·      ·    1/3      ·    2/3      ·      ·      ·
mat            ·      ·      ·      ·      ·      ·      ·      ·      ·      ·      1
on             ·      ·      ·      ·      ·      ·      ·      ·      1      ·      ·
park           ·      ·      ·      ·      ·      ·      ·      ·      ·      ·      1
ran            ·      ·      ·      ·      ·      ·      ·      ·      ·      1      ·
rug            ·      ·      ·      ·      ·      ·      ·      ·      ·      ·      1
sat            ·      ·      ·      1      ·      ·      ·      ·      ·      ·      ·
the          1/4    1/4    1/6      ·    1/6      ·    1/6      ·      ·      ·      ·
to             ·      ·      ·      ·      ·      ·      ·      ·      1      ·      ·
```

**Zero-probability transitions**

| from | zero-probability next tokens |
|---|---|
| `the` | on, ran, sat, the, to, `<END>` (6 of 11) |
| `cat` / `dog` | cat, dog, mat, on, park, rug, the, to, `<END>` (9 of 11) |
| `sat` | everything except `on` (10 of 11) |
| `ran` | everything except `to` (10 of 11) |

- **104 of the 121 entries (86%) are zero.** Only 17 distinct transitions were ever observed.
- Some zeros are **genuinely impossible** in English (`the the`, `the <END>`, `cat the`).
- Others are **grammatical but unseen**: `cat on` ("the cat on the mat"), `ran on`, `dog <END>`. Maximum-likelihood estimation cannot tell "impossible" from "not seen yet", and any sentence containing such a pair gets joint probability 0.
- The converse also happens: `<START> → the → mat → <END>` is a non-zero path, so the model accepts the fragment "the mat".
- `<END>` never appears as a context. It is absorbing, which is what stops generation.

---

## Question 4: Where are the transition counts stored?

In `self.counts`, a `defaultdict(Counter)` created in `FirstOrderLM.__init__` and filled in `fit`. `self.counts[prev][nxt]` is $C(\text{prev}, \text{nxt})$. It is incremented once per consecutive pair from `zip(tokens[:-1], tokens[1:])`, which covers every transition, including `<START> → the` and `mat → <END>`.

## Question 5: Where is $P(X_t \mid X_{t-1})$ computed?

At the end of `fit`, in the normalisation loop. For each context the row total $\sum_k C(\text{ctx}, w_k)$ is computed and every count is divided by it, giving `self.probs[ctx][w]`. `distribution(ctx)` only looks the row up, so probabilities are computed once, not on every call.

## Question 6: How does the program choose the next word?

It supports both methods, selected by `mode` in `generate`:

1. **Greedy (`mode="greedy"`)** calls `predict`, which always takes the argmax of P(w | ctx) (ties broken alphabetically).
2. **Sampling (`mode="sample"`, the default)** calls `sample_next`, which draws w ~ P(· | ctx) using `rng.choices(words, weights=probs)`.

**The difference:** greedy is **deterministic**. The same context always gives the same word, so it can only ever produce one sentence and it ignores every lower-probability alternative. Sampling is **stochastic**: a word with probability 1/3 is chosen about a third of the time, so repeated runs give different sentences in proportion to their probability. Only sampling draws from the joint distribution the Bayesian network defines. Greedy returns one locally-best path, which need not be the most probable sentence and, here, never even finishes (Question 10).

## Question 7: What happens with a word for which no transition has been observed?

There are two cases:

- **Unseen context** (an out-of-vocabulary word like `bird`, or `<END>`). There is no CPT row, so P(· | bird) is undefined (0/0). The implementation returns an empty distribution, `predict`/`sample_next` return `None`, and `generate` stops with `finished=False`. A naive `probs[word]` would raise `KeyError`. For `<END>` this is exactly the intended behaviour.
- **Seen context, unseen transition** (e.g. `cat → on`). The entry is 0, so that word can never be generated after `cat`, and any sentence containing the pair gets probability 0.

The standard remedies, not used here so that the CPT stays the pure maximum-likelihood estimate:
- **add-k (Laplace) smoothing**: $(C(v,w)+k)/(C(v)+k|V|)$;
- **back-off or interpolation** with a lower-order (unigram) model;
- an **`<UNK>` token** for out-of-vocabulary words.

---

## Question 8: If one row sums to 0.87, what does that tell you?

The implementation is **wrong**, not just imprecise. Floating-point error is on the order of $10^{-16}$, not 0.13. That row is not a probability distribution: 13% of the probability mass has gone missing, because the counts and the normaliser disagree. Typical causes:

- **The denominator counts more than the numerators.** Examples: dividing by the unigram count C(w), including positions where w has no successor (sentence-final words when `<END>` is not added); or smoothing the denominator (+k|V|) without adding k to each numerator.
- **Transitions are dropped while counting.** Examples: an off-by-one loop `range(len(tokens) - 2)` that skips the last pair (`mat → <END>`); or successors filtered out of the row but still counted in the total.
- **Truncation or rounding** of stored probabilities.

This bug is easy to miss because `random.choices` silently renormalises its weights, so sampling still appears to work. Every sentence probability and every model comparison would still be wrong. The notebook builds a deliberately buggy variant (smoothing applied to the denominator only). Its rows sum to between 0.65 and 0.92, and the normalisation test flags every one of them as `FAIL`. The real model's rows all sum to 1.000000000000.

---

## Question 9: Are the most probable predictions what you would expect?

| context | argmax | full distribution |
|---|---|---|
| `<START>` | the | the 1.0 |
| `the` | cat (tie with dog) | cat .25, dog .25, mat .167, rug .167, park .167 |
| `cat` / `dog` | sat | sat .667, ran .333 |
| `sat` | on | on 1.0 |
| `ran` | to | to 1.0 |
| `on` / `to` | the | the 1.0 |
| `mat` | `<END>` | `<END>` 1.0 |

The predictions are only partly what a person would expect. `sat → on`, `ran → to` and `mat → <END>` match intuition, but several don't:

- **After `the`, the argmax is a tie** between `cat` and `dog`. A person reading "the cat sat on **the** …" expects `mat` or `rug`. The model sees only the single word `the` and cannot tell subject position from object position, so it predicts `cat` there too. It also gives `park` the same weight as `mat`, even though "sat on the park" is odd.
- **After `sat`, the model is certain** (`on`, probability 1). A person would also accept "sat down", "sat quietly", or the end of the sentence. The model treats everything it has not seen as impossible.
- **Where the model agrees with a person** (for example, "the sat" is impossible), it does so only because the pair did not occur, not because it knows any grammar.

**Probability model vs human expectations:** a probability model summarises the *frequencies in its training data* under its structural assumption (here, one word of context). It has no semantics, world knowledge or grammar beyond what the counts encode, and its "expectation" is relative to a six-sentence corpus. Human expectations draw on the whole sentence, on meaning, and on vastly more linguistic experience, and they rarely assign exactly 0 or 1. The model can be a perfectly correct estimator of $P(X_t \mid X_{t-1})$ and still disagree with a person, because it answers a narrower question. And the argmax is only one point: the model's real prediction is the whole distribution.

---

## Question 10: Greedy vs sampling: which produces more variation, and why?

Five sentences from each mode (first-order model):

```
Greedy                                                           Sampling
the cat sat on the cat sat on the cat sat on the cat sat on …    the rug
the cat sat on the cat sat on the cat sat on the cat sat on …    the rug
the cat sat on the cat sat on the cat sat on the cat sat on …    the cat ran to the dog ran to the park
the cat sat on the cat sat on the cat sat on the cat sat on …    the cat sat on the dog sat on the dog sat on the cat sat on the mat
the cat sat on the cat sat on the cat sat on the cat sat on …    the cat sat on the mat
```

**Sampling produces all the variation. Greedy produces the same output five times, and that output is not even a sentence.**

- Greedy is a deterministic function of the context. Starting from the same `<START>` with the same CPT, it must follow the same path every time, so only one output is possible.
- That path is `the → cat → sat → on → the → cat → …`, a **cycle**. `<END>` can only follow `mat`/`rug`/`park`, and none of those is ever the argmax after `the` (1/6 < 1/4). So greedy decoding never terminates and has to be cut off by `max_tokens`. A first-order model has no memory that it has already said "the cat".
- Sampling takes lower-probability branches in proportion to their probability. It reaches `<END>` with probability 1 and produces a different sentence on almost every run (four distinct sentences out of five above, from 2 to 18 words). The variation comes from the entropy of the CPT rows: deterministic rows like `sat → on` contribute none, and `the` contributes the most.

Greedy maximises each step locally, which is not the same as finding the most probable sentence. Under this model the most probable complete sentences are actually the fragments "the mat" / "the rug" / "the park" (P = 1/6 each). "the cat sat on the mat" only gets 1 · 1/4 · 2/3 · 1 · 1 · 1/6 · 1 = 1/36. Greedy produces neither of them.

For contrast, the **second-order** model's greedy output does terminate, with "the cat sat on the mat", because the context `on the` leads to `mat`/`rug` rather than back to `cat`.

---

## Question 11: How does the second-order model differ from the first-order model?

| | First-order | Second-order |
|---|---|---|
| **1. Graph structure** | chain; each node has one parent, $X_{t-1}\to X_t$ | adds a skip edge; each node has two parents, $X_{t-2}\to X_t \leftarrow X_{t-1}$ |
| Independence assumption | $X_t \perp X_1..X_{t-2} \mid X_{t-1}$ | $X_t \perp X_1..X_{t-3} \mid X_{t-2}, X_{t-1}$ (weaker) |
| **2. CPT** | rows indexed by one word; estimated from pair counts C(v, w) | rows indexed by a *pair* of words; estimated from triple counts C(u, v, w) |
| CPT size / free parameters (here) | 11 contexts × 11 = 121 / 110 | 111 contexts × 11 = 1221 / 1110 |
| **3. Context for prediction** | one word, so all three uses of `the` share one row | two words, so it separates `<START> the` (cat/dog), `on the` (mat/rug) and `to the` (park) |
| **4. Data needed** | little; all 11 contexts were observed | about 10× the parameters from the same data; 96 of 111 contexts were never observed |

---

## Question 12: Why does more context help prediction, and why does it make estimation harder?

Measured comparison of the two models:

| | first-order | second-order |
|---|---|---|
| Full CPT size / free parameters | 121 / 110 | 1221 / 1110 |
| Non-zero entries actually estimated | 17 | 19 |
| Deterministic rows (p = 1) | 8 of 11 | 11 of 15 |
| Unseen (zero-probability) contexts | 0 of 11 | 96 of 111 |
| Distinct sentences in 1000 samples | 154 | 6 |
| Novel sentences (not in training data) | 148 | 0 |
| Mean / max sentence length | 5.78 / 42 words | 6.0 / 6 words |
| P("the cat sat on the mat") | 0.028 | 0.167 |
| P("the cat ran to the mat"), plausible but unseen | 0.014 | **0** |

**Why more context helps.** Conditioning on more of the past can only reduce uncertainty on average ($H(X_t \mid X_{t-1}, X_{t-2}) \le H(X_t \mid X_{t-1})$), and it moves the model closer to the true chain-rule factor $P(X_t \mid X_1, \dots, X_{t-1})$. Here the extra word separates the three uses of `the`. That removes the fragments ("the park"), the wrong prepositions ("sat on the park") and the greedy loop, and it raises each training sentence's probability from about 1/36 to 1/6.

**Why it makes estimation harder.** The CPT grows exponentially with the context length k: $|V|^k$ rows and $|V|^k(|V|-1)$ free parameters (110 → 1110 here, with only 10 words). The number of training tokens N is fixed, so the average number of observations per context falls like $N/|V|^k$. As a result:

- **most contexts are never observed** (96 of 111), so the model has no prediction at all for them;
- **observed contexts are seen only once or twice**, so their estimates are extreme (mostly probability 1) and high-variance;
- **valid but unseen continuations get probability 0** ("the cat ran to the mat"), so the model memorises the training set instead of generalising. Every sentence it generates is one of the six training sentences.

Notice that the number of *estimated* parameters barely moves (17 → 19) while the table grows 10×. Six sentences simply do not contain more patterns than that, so the extra capacity goes almost entirely into empty rows. This is a bias–variance trade-off:

- **Short context:** a biased model (the independence assumption is wrong) with well-estimated parameters.
- **Long context:** a less biased model with poorly estimated parameters.

N-gram models manage this trade-off with smoothing and back-off. Neural language models manage it by sharing parameters across contexts.

---

## Question 13: Why is Approach B preferable to Approach A?

- **Approach A:** "Write a Python language model for me." This hands the modelling decisions to the LLM.
- **Approach B:** "Implement P(X_t | X_{t-1}), estimated from transition counts, with sampling-based generation." The modelling decisions stay with you, and the LLM is only asked for code.

Why B is better:

- **It specifies the intended behaviour.** B pins down the variables, the dependency structure, the estimator and the generation procedure. With A the LLM could return anything from a bigram model to a call to a pretrained transformer, and nobody could say whether it is "correct", because correct was never defined.
- **You understand the representation.** Knowing the model is a CPT of normalised bigram counts tells you where to look in the code (`self.counts`, `self.probs`) and what the numbers must be (P(cat | the) = 3/12). That is what made Questions 4–7 answerable.
- **The implementation can be validated.** A specification can be checked against. Comparing the generated code to hand-computed counts exposed behaviour the spec had not covered: greedy mode never terminating, ties broken silently by dictionary order, and a `KeyError` on unseen words. Those cases were then added to the spec.
- **You can test probabilistic invariants.** B implies properties that must hold however the code is written:
  - every row sums to 1, with all probabilities in [0, 1];
  - the counts add up to the number of transitions;
  - a sentence's probability equals the product of its factors.

  These catch real bugs, like the 0.87 row in Question 8, that looking at outputs would not, because `random.choices` hides a non-normalised table.
- **Implementation is kept distinct from model.** The *model* is the Bayesian network and its CPT; the *implementation* is one of many programs that could realise it. B keeps the two separate. The second-order change was stated as a change to the model (two parents, triple counts) before any code was accepted, and the same tests validated both implementations. A merges model and implementation, so a coding bug cannot be told apart from a modelling flaw.

---

## Question 14: What did thinking of the language model as a Bayesian network give you?

- **A representation of dependencies.** The graph $X_{t-1} \to X_t$ makes explicit what each word may depend on. Adding one edge ($X_{t-2} \to X_t$) *is* the step from first- to second-order, and everything else (the CPT shape, what must be counted) follows from the graph.
- **A factorisation of the joint distribution.** The Bayesian network writes $P(X_1, \dots, X_T)$ as a product of local CPTs. That gives sentence probabilities directly, and it explains why a single unseen bigram makes a whole sentence's probability 0.
- **A way to interpret conditional probabilities.** Every CPT entry has a concrete meaning ("the fraction of times `the` was followed by `cat`"), so predictions can be read and questioned directly. For example: why does `park` get 1/6 after "sat on the"?
- **A principled method for generation.** Ancestral sampling in topological order is the standard way to sample from a Bayesian network, and it draws exactly from the joint distribution. It also explains why greedy decoding is *not* sampling from the model, and why it can loop.
- **A way to reason about independence assumptions and the effect of context.** The Markov assumption explains the incoherent first-order sentences. Counting CPT parameters ($|V|^k(|V|-1)$) explains why the second-order model memorised the data.
- **A way to test whether an implementation matches its specification.** Every CPT row must be a distribution, so normalisation is a property test that follows directly from the definition of a Bayesian network. It validated both LLM-assisted implementations, and it catches the 0.87 bug from Question 8.
