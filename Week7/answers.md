# Week 7: Answers — Transformers and Running GPT-type Models Locally

**Lab material:** [Transformer.pdf](../lab-question/Transformer.pdf), [AI_lab_transformers.ipynb](../lab-question/AI_lab_transformers.ipynb), [Run_Ollama.ipynb](../lab-question/Run_Ollama.ipynb)
**Solutions:** [Transformers_Lab.ipynb](../solution/Transformers_Lab.ipynb) (Part 1) and [Ollama_LLM_Comparison.ipynb](../solution/Ollama_LLM_Comparison.ipynb) (Part 2)

The notes have no numbered questions. This file answers each topic the notes set out: what the Transformer was introduced for, the encoder-only and decoder-only families, the architecture, the six concepts, the three hands-on tasks, and the Ollama comparison with GPT and Claude. Every result below comes from running the models.

---

## 1. What was the Transformer initially introduced for?

**Machine translation**, and more generally **sequence-to-sequence** tasks: reading one sequence and producing another. It was introduced in *Attention Is All You Need* (Vaswani et al., 2017) as an **encoder-decoder**:
- the encoder reads the whole source sentence;
- the decoder generates the target one token at a time, attending to its own output so far and, through cross-attention, to the encoder.

**In this lab:**
- bert2bert translated "I love machine learning." → "Ich liebe maschinelles Lernen.";
- T5-base translated it into French ("J'aime l'apprentissage par machine."), German and Romanian.

## 2. Encoder-only Transformers (e.g. BERT)

They use **bidirectional** self-attention: every token sees the whole input. They are trained to **understand and represent** text, not to generate it. Typical uses are semantic search, document retrieval, clustering and classification.

**Evidence from the lab:**
- **Semantic search** (MiniLM sentence embeddings): **6 of 6** reworded queries found the right FAQ, for example "money back if I cancel" → "refund policy for cancelled orders". Keyword matching found **0 of 6**.
- **Bidirectional context** (BERT fill-mask): "I went to the [MASK] to deposit money." → **bank** (0.87). The clue comes *after* the gap. "She sat on the bank of the [MASK]." → **river**.
- **Sentiment analysis** (DistilBERT, the lab's default): correct on clear cases and simple negation, but only **7 of 11** probes with a definite answer. It failed on double negation, both **sarcasm** probes (POSITIVE, 0.98) and "The tumour shrank…" (NEGATIVE, 0.999), with 0.99 mean confidence when wrong. Neutral sentences are forced into POSITIVE or NEGATIVE, because the model only has two labels.

## 3. Decoder-only Transformers (e.g. GPT)

They use **causal** self-attention: token $t$ sees only tokens $\le t$. They are trained to predict the **next token**, and are used for **generation**: text, chatbots, code, question answering, summarisation and writing assistance.

**Evidence from the lab (GPT-2):**
- **The lab's call ignored `max_length=30`.** The `text-generation` pipeline's default `max_new_tokens=256` took precedence, so it generated **261 tokens**, including an invented quotation.
- **Decoding strategies:**
  - greedy decoding **loops** ("The future of AI is uncertain." repeated);
  - beam search is fluent but repetitive;
  - `no_repeat_ngram_size=3` removes the loop;
  - temperature trades diversity for coherence: T = 0.7 stays coherent, while T = 1.5 drifts.
- **Next-token distributions:** after "The capital of France is", " Paris" is only the **5th** most likely token (3.2%). GPT-2 predicts how text continues; it does not answer questions.

## 4. The architecture (the diagram in the notes)

1. **Input embeddings plus positional encodings** feed a stack of **N encoder layers**. Each layer has multi-head **self-attention**, add & norm, a feed-forward network, and add & norm again.
2. **The target sequence, shifted right,** is embedded, given positional encodings, and passed through **N decoder layers**. Each decoder layer has **masked** multi-head self-attention, then multi-head **cross-attention** (queries from the decoder, keys and values from the encoder's output), then a feed-forward network, each followed by add & norm.
3. **A final linear layer and softmax** produce the probability distribution over the next token.

The recorded models share this block design. BERT-base, GPT-2 and each half of T5-base all have 12 layers, 12 heads and a hidden size of 768; bert2bert has 24 + 24 layers, 16 heads and 1024. Sizes range from 23M parameters (MiniLM) to 772M (bert2bert).

## 5. The six concepts

Each concept was implemented from scratch in NumPy and checked against PyTorch.

| Concept | What it is | Demonstrated |
|---|---|---|
| **Attention mechanism** | $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$: each position takes a weighted average of values, with weights from query–key similarity | matches `F.scaled_dot_product_attention` to $4\times10^{-16}$. Without the $\sqrt{d_k}$ scaling, at $d_k = 1024$ the softmax saturates (mean max weight 0.96); with it, attention stays soft (0.36). |
| **Self-attention** | $Q$, $K$, $V$ all projected from the *same* sequence | in BERT, "it" attends to "animal" more than to "street" in **93 of 144 heads**; layer 3, head 10 puts **0.85** on "animal" |
| **Cross-attention** | $Q$ from the decoder, $K$ and $V$ from the encoder, giving a (target × source) matrix | in T5, *liebe* → *love*, *maschine* → *machine*, *Lern* → *learning*. The largest weight goes to the task token "German", so a raw attention map is not a clean alignment. |
| **Multi-head attention** | $h$ parallel attentions on lower-dimensional projections, concatenated and mixed by $W_O$ | two heads on the same input give different patterns. Real heads specialise, for example the pronoun-resolving heads in BERT. |
| **Positional encoding** | position information added to the embeddings, because attention alone is order-blind | self-attention without positions is **permutation-equivariant** (verified), and adding sinusoidal encodings breaks that. $PE(p)\cdot PE(p+k)$ depends only on the offset $k$. |
| **Masked attention** | future positions set to $-\infty$ before the softmax | changing a future token leaves all earlier outputs unchanged (verified, and it matches PyTorch's `is_causal=True`). In GPT-2 the weight above the diagonal is exactly **0.0** in every layer and head, while BERT's reaches 1.0. |

**Also observed: attention sinks.** In GPT-2's last layer, about 0.71 of the attention goes to the first token; in BERT, about a third goes to `[SEP]`. Attention weights are not simple "importance" scores.

## 6. The hands-on session

**Machine translation (encoder-decoder):**
- **bert2bert:** the lab's `add_special_tokens=False` is the correct call for this model. Default tokenisation produced "Pflanzen **Pflanzen**…", "**i** liebe…" and "**die** Wetter".
- **The lab's own input is a half-sentence.** With the lab's call it became "a *so-called* process"; with default tokenisation the model invented "known as *conservation*".
- **T5 word-sense errors:** "Plants" → "Utilităţile" ("utilities") in Romanian, and "today" → "heutzutage" ("nowadays") in German.
- **T5 only knows its trained languages:** asked for Spanish it outputs **German**, and asked for Hindi it outputs "Hindi Ich liebe…".
- **The Hinglish model** reproduces the lab's "Life crazy haiNot me" exactly.

**Sentiment analysis (encoder-only):** see Section 2.

**GPT-type generation (decoder-only):** see Section 3.

## 7. Running GPT-type models locally with Ollama, compared with GPT and Claude

**Setup:**
- Ollama 0.17.5 was installed with the official installer, and `mistral` (7.2B, 4-bit Q4_K_M, a 4.4 GB download) was run on an Apple-silicon Mac.
- The lab's LangChain cell ran unchanged and gave a correct Newton biography. The lab's own Colab run had failed at `ollama list` (`command not found`), because the install step inside `%xterm` had not been run.
- 12 fixed prompts with reference answers were given to Mistral, to **GPT 5.6 Sol** and **Claude Sonnet 5.5** (in their apps), and to four open-weight models from 2.6B to 120B parameters.
- Each answer was graded, and each model's `is_prime` was executed against 200 test values.

| Model | Score (of 12) | Wrong or flawed |
|---|---|---|
| **Mistral-7B (local, Ollama)** | **8.5** | 347 × 29 = **10189** (it is 10063); **invented a summary of a non-existent paper**; listed speech recognition, translation and summarization as encoder-only uses; one wrong date (Royal Society presidency). It followed only 4 of 8 format instructions. |
| **GPT 5.6 Sol** | **12** | none (web search was on for the fake-paper prompt) |
| **Claude Sonnet 5.5** | **12** | none |
| LFM-2.5-2.6B | 8.5 | ran out of tokens on hidden reasoning; invented the paper summary (when given more tokens); invalid encoder-only uses; "Je aime" |
| Qwen3.8-27B | 12 | none |
| Gemma-4-31B | 11 | **invented the paper summary** |
| Nemotron-3-Super-120B | 11.5 | correct refusal, but claimed a database search it cannot perform, and invented an author profile |

**Compared with GPT and Claude, the local Mistral-7B:**
- **matches them** on facts, the bat-and-ball trap, the word problem, working `is_prime` code (all seven models passed 200/200), French translation and sarcasm;
- **falls behind** on exact arithmetic, **admitting it doesn't know** (it confidently invents content), precise knowledge, and following format instructions;
- **offers** privacy, offline use, zero cost and reproducibility, at about **13.8 tokens per second**, after an 11 s model load.

**Size is not the whole story.** Gemma-4-31B invented the fake paper while Qwen3.8-27B declined. Refusing to answer an unanswerable question comes from training, not from size.

**Three architectures, one sarcastic sentence:**
- DistilBERT, a 67M encoder classifier: **POSITIVE**;
- all seven chat models, which are decoder-only: **NEGATIVE**.

**The takeaway:** every model in Part 2 is a decoder-only Transformer predicting $P(\text{next token} \mid \text{previous tokens})$, the same objective as GPT-2 in Part 1 and the bigram model in Week 8. What separates them is scale, training, instruction tuning and quantisation. Their errors sound as confident as their correct answers, so outputs have to be checked against references, and code has to be run.
