# Week 6: Answers to the Lab Exercises

**Lab:** Building and Learning a Bayesian Network ([lab notebook](../lab-question/llm_bn.ipynb))
**Solution with all code and outputs:** [solution/LLM_BN_Solution.ipynb](../solution/LLM_BN_Solution.ipynb)

The network, used throughout: $C \to R,\ C \to S,\ R,S \to W$, with $P(C,R,S,W) = P(C)\,P(R\mid C)\,P(S\mid C)\,P(W\mid R,S)$.

The parameters:

| CPT | Entries |
|---|---|
| $P(C{=}1)$ | 0.5 |
| $P(R{=}1\mid C)$ | 0.2 when $C=0$, 0.8 when $C=1$ |
| $P(S{=}1\mid C)$ | 0.5 when $C=0$, 0.1 when $C=1$ |
| $P(W{=}1\mid R,S)$ | 0.01, 0.90, 0.90, 0.99 for $(R,S) = (0,0), (0,1), (1,0), (1,1)$ |

**Models used:**
- **Qwen2.5-Coder-1.5B-Instruct** (the lab's model): the main workflow.
- **Qwen2.5-Coder-32B-Instruct** and **Qwen3-Coder-30B-A3B-Instruct**: the size comparison.
- **Qwen2.5-Coder-32B-Instruct** also answered the query and sampling-variability exercises.

All requests used the lab's system prompt and temperature 0.

---

## Part 1: Validating the LLM-generated programs (the lab's approval and validation cells)

In the lab notebook every approval flag was left at `False`, so the generated code was never run. Here each program was checked, executed, and validated against a trusted reference: a hand-built model, brute-force enumeration of all 16 joint probabilities, and the trusted `DiscreteMLE` / BDeu fits.

| Task | Round | Qwen2.5-Coder-1.5B result |
|---|---|---|
| Inference | 1 (lab prompt) | **fails**: removed `BayesianNetwork` class, CPDs with no parents declared, Rain rows reversed, Sprinkler not normalised, WetGrass rows swapped, nonexistent `model.query` |
| | 2 (feedback on each failure) | **fails**: right class and query now, but the edges were dropped, `VariableElimination` is not imported, and every CPT is unchanged |
| | 3 (construction template) | **fails**: copied the template literally, mixing unrelated numbers into one list |
| | human fix | five targeted edits to round 2 → **PASS**: structure ✓, `check_model` ✓, all 16 joint probabilities match, $P(R{=}1\mid W{=}1) = 0.70477$ |
| MLE estimation | 1 | **fails**: removed class, *invents its own data* (overwriting the real `data`), `pd` not imported |
| | 2 (feedback) | **PASS**: every CPT entry equals the trusted MLE fit exactly |
| BDeu estimation | 1 | **fails**. The lab's recorded version calls `pd.read_csv(...)`, a forbidden file read that the lab's own AST check misses (it only inspects `name(...)` calls, not `obj.method(...)` calls). The new version uses a removed class and a nonexistent `.fit()`. |
| | 2 (feedback) | **PASS**: equals the trusted BDeu fit, and is verified to differ from MLE and from ESS = 50, so the requested prior is really used. (In the local run this round **failed**; see Part 6.) |

**Does a bigger model fix it?** The lab's original three prompts were sent to all three models, and **none solved any task.**
- **Qwen2.5-Coder-1.5B fails loudly:** every failure is an exception.
- **The two larger models fail quietly on inference.** Their code uses the correct API, runs, and passes `check_model()`, but the posterior is wrong:
  - **32B:** Sprinkler wrong and WetGrass rows swapped, giving $P(R{=}1\mid W{=}1) = 0.40$.
  - **Qwen3-30B:** Sprinkler scrambled, giving 0.56.

  Only semantic tests caught these: the joint-distribution and posterior checks.
- **On estimation, the larger models failed on API versions:** `fit(data, estimator=MaximumLikelihoodEstimator)` is rejected by pgmpy 1.1.2, `get_parameters()` returns a list and was treated as a model, and removed classes were imported.

**Reproducibility.** None of the three responses to the lab's prompts matched the responses recorded in the lab notebook, even with the same model, prompt and greedy decoding. Across repeated runs, the estimation prompt produced three different programs. A generated program must therefore be saved as an artifact; re-running the prompt does not reproduce it.

---

## Part 2: HW exercise, "change the query"

**Before running: the query variable, the evidence, and a qualitative prediction.**

Priors: $P(S{=}1) = 0.30$, $P(C{=}1) = 0.50$, $P(R{=}1) = 0.50$. For comparison, $P(R{=}1\mid W{=}1) = 0.705$.

| Query | Query variable | Evidence | Prediction | Reasoning |
|---|---|---|---|---|
| $P(S{=}1\mid W{=}1)$ | Sprinkler | WetGrass = 1 | increases from 0.30, but stays below 0.5 | observing an effect raises belief in each of its causes; rain is the more common cause |
| $P(C{=}1\mid W{=}1)$ | Cloudy | WetGrass = 1 | slight increase from 0.5 | wet grass → rain more likely → cloudy more likely. But wet grass → sprinkler more likely → *not* cloudy. The rain path is stronger, since 0.8 vs 0.2 is a bigger contrast than 0.5 vs 0.1. |
| $P(R{=}1\mid W{=}1, S{=}0)$ | Rain | WetGrass = 1, Sprinkler = 0 | sharp increase, near 1 | with the sprinkler off, rain is almost the only explanation, since $P(W{=}1\mid R{=}0,S{=}0) = 0.01$. This is the reverse of explaining away. |

**Generated code, inspected and compared with a trusted `VariableElimination` call and with enumeration.** The LLM was given the validated model as `model` and asked only for the query.

| Model | Query | Generated call | Generated | Trusted VE | Enumeration | Match |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | $P(S{=}1\mid W{=}1)$ | `ve.query(variables=['Sprinkler'], evidence={'WetGrass': 1})` | 0.427846 | 0.427846 | 0.427846 | ✓ |
| Qwen2.5-Coder-32B | $P(S{=}1\mid W{=}1)$ | `inference.query(variables=['Sprinkler'], evidence={'WetGrass': 1})` | 0.427846 | 0.427846 | 0.427846 | ✓ |
| Qwen2.5-Coder-32B | $P(C{=}1\mid W{=}1)$ | `inference.query(variables=['Cloudy'], evidence={'WetGrass': 1})` | 0.574615 | 0.574615 | 0.574615 | ✓ |
| Qwen2.5-Coder-32B | $P(R{=}1\mid W{=}1,S{=}0)$ | `inference.query(variables=['Rain'], evidence={'WetGrass': 1, 'Sprinkler': 0})` | 0.992202 | 0.992202 | 0.992202 | ✓ |

**All three predictions held:** 0.4278 (up from 0.30, below 0.5), 0.5746 (a slight rise from 0.5), and 0.9922 (up from 0.705). **Every query in this table was correct**: right query variable, right evidence, and a `VariableElimination` call on the existing model. The lab's 1.5B model run *locally*, however, got all three wrong (Part 6).

The contrast with Part 1 is the lesson. A one-line library call on an already-validated model is well within what the LLM does reliably. Transcribing a probability table into a positional array convention is not. The more of the scientific content the LLM has to encode, the more its output needs validating.

---

## Part 3: Classroom exercise, "let the LLM explain its generated CPT"

**Question asked:** *Explain exactly what each column of the WetGrass `TabularCPD` represents when `evidence=["Rain", "Sprinkler"]`. Do not change the code.*

**The actual `pgmpy` convention**, verified by reading each column back with `get_value`:
- columns are parent configurations, with the **last** evidence variable changing fastest: $(R,S) = (0,0), (0,1), (1,0), (1,1)$;
- rows are the child's states: row 0 is $W=0$, row 1 is $W=1$;
- so each column is a full distribution $P(W\mid R{=}r,S{=}s)$ and sums to 1.

The specified values 0.01, 0.90, 0.90 and 0.99 sit in row 1 of columns 1–4, as intended.

**The two explanations:**
- **Qwen2.5-Coder-1.5B is wrong.** It calls the columns "WetGrass=0" and "WetGrass=1", confusing columns with rows. It assigns the same evidence ("Rain=1 and Sprinkler=1") to different columns, and it reads 0.99 as "a high probability of not being wet" when both causes are present. The model that could not *write* the convention could not *explain* it either.
- **Qwen2.5-Coder-32B is correct.** It gives the right column order, says that each row is a state of WetGrass, and that each column is a distribution. Yet the same 32B model wrote a wrong Sprinkler table in Part 1.

A correct explanation does not guarantee correct generation, so the two have to be checked separately, here against `get_value`.

---

## Part 4: Classroom exercise, "estimate from different datasets"

**The lab's cell** ($N = 100$, seeds 1–5), which had no output in the lab notebook:

| Seed | Rows with Cloudy = 1 | $\widehat{P}(R{=}1\mid C{=}1)$ | Abs. error vs 0.8 |
|---|---|---|---|
| 1 | 49 | 0.6939 | 0.1061 |
| 2 | 44 | 0.7955 | 0.0045 |
| 3 | 47 | 0.7447 | 0.0553 |
| 4 | 56 | 0.7679 | 0.0321 |
| 5 | 49 | 0.7959 | 0.0041 |

**Extended to 500 datasets per size:**

| $N$ | Mean estimate | Std of estimate | Theory $\sqrt{0.8\cdot0.2/(N/2)}$ |
|---|---|---|---|
| 50 | 0.7941 | 0.0854 | 0.0800 |
| 100 | 0.8041 | 0.0570 | 0.0566 |
| 500 | 0.7997 | 0.0248 | 0.0253 |
| 2000 | 0.8002 | 0.0127 | 0.0126 |

**The scientific explanation is sampling variability.** Each dataset is a different random sample from the same network. The MLE is a relative frequency computed only from the rows with Cloudy = 1 (about $N/2$ of them), so it fluctuates like a binomial proportion:
- it is centred on the true 0.8 (unbiased);
- its standard deviation is $\sqrt{p(1-p)/N_{C=1}}$, which shrinks as $1/\sqrt{N}$;
- the measured spread matches this prediction almost exactly.

Neither the LLM nor pgmpy is inconsistent: identical data gives identical estimates.

**The LLM's explanation (Qwen2.5-Coder-32B)** correctly attributed the differences to sampling variability: different random samples give different observed frequencies, and so different MLEs. It was correct but only qualitative. It did not mention that only about 50 rows inform this parameter, or that the spread shrinks as $1/\sqrt{N}$. The measurements supply those points.

---

## Part 5: "Where could the system fail?" (observed, not hypothetical)

| Failure mode listed in the lab | Observed? |
|---|---|
| Ambiguous specification | **Yes.** The prompts said "use MaximumLikelihoodEstimator" but gave no API version. The models used calling conventions the installed pgmpy rejects, and the lab's own trusted code already uses `DiscreteMLE`. |
| Obsolete API | **Yes, in every model:** `BayesianModel`, `BayesianNetwork`, `model.query`, `fit(estimator=Class)` |
| Reversed CPT row or parent ordering | **Yes:** rows reversed or swapped (1.5B, 32B), a scrambled table (Qwen3-30B), and parents never declared (1.5B) |
| Unintended imports or calls | **Yes:** `pd.read_csv` despite "do not read files", which the lab's AST check missed |
| Code runs but represents the wrong distribution | **Yes:** the 32B and Qwen3-30B inference programs |
| Wrong evidence state in the query | No; all the generated queries were correct |
| Dataset encoding differs from the model | No, but the generated code *replaced* the dataset with invented rows |
| Sparse data make estimates unstable | Yes, by design: at $N = 100$ the estimates range from 0.69 to 0.80 |
| Undocumented prior | No; the BDeu prior was verified explicitly |
| Validation checks syntax but misses semantics | **Yes:** `check_model()` accepted two wrong models |

**What the LLM contributed:**
- program skeletons for every task;
- correct MLE code after one round of concrete feedback, in both the hosted and the local run (BDeu only in the hosted run);
- correct one-line queries from 32B, but not reliably from 1.5B;
- one correct explanation (32B).

**What it did not contribute:** a single correct probability table, from any model at any size. It never computed a probability, either. `pgmpy` did every computation, and the trusted reference, the enumeration oracle, the semantic tests and the human fixes decided what was correct.

$$\text{program runs} \;\neq\; \text{probabilistic model is correct} \;\neq\; \text{scientific assumptions are appropriate}$$

---

## Part 6: The lab's model run locally, exactly as the lab notebook does

All 12 prompts were replayed on **Qwen2.5-Coder-1.5B-Instruct** loaded locally: a `transformers` pipeline, `do_sample=False`, `max_new_tokens=1400`, on an Apple GPU. The responses are in [solution/outputs/local_responses.json](../solution/outputs/local_responses.json). Every response went through the same static check, execution and validation as the saved ones.

| Code task | Local run | Saved (hosted) run |
|---|---|---|
| Inference, rounds 1–3 | all fail | all fail |
| MLE, round 1 → 2 | fail → **pass** | fail → **pass** |
| BDeu, round 1 → 2 | fail → **fail** (still calls `pd.read_csv`) | fail → pass |
| The three HW queries | **0 / 3** (removed `BayesianModel`, and the provided `model` overwritten) | 1 / 1 |
| **Total passing** | **1 of 10** | **3 of 8** |

- **Local runs reproduce each other.** Two of the three original prompts gave *exactly* the same response as the lab notebook's recorded run. None of the 12 matched the saved hosted responses, whose similarity ranged from 0.10 to 0.78, so the serving setup is what changed the output.
- **The explanations:**
  - The local CPT explanation has the right column order, but describes each column as only $P(W{=}0 \mid \cdot)$, when a column holds the whole distribution.
  - The local variability explanation correctly blames sampling randomness and the small sample size. But it wrongly says seeded simulations are "not perfectly replicable".
- **Speed:** 15–42 s per response.

**Did running through an API cause the errors? No.** Run locally, the same model made every category of error seen through the API (removed classes, wrong CPT layouts, invented data, forbidden file reads), and it passed *fewer* tasks. The serving path changed *which* wrong program came out, not *whether* it was wrong. Only MLE-after-feedback is robust across both runs.

