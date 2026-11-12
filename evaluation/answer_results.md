# Answer evaluation

20 questions answered by the full workflow with `qwen/qwen3.8-27b` (tree index, chunks of 1000 / 150). The same model judged the answers.

- **Correct** (same fact as the reference answer): 20 of 20
- **Grounded** (every claim is in the retrieved passages): 18 of 20
- **Cited** (the answer points to a numbered passage): 20 of 20
- **Searched twice** (the grader asked for a rewritten question): 0 of 20

| # | Question | Correct | Grounded | Cited |
|---|---|---|---|---|
| 1 | How many parallel attention heads does the Transformer use? | yes | yes | yes |
| 2 | How long did it take to train the base Transformer model? | yes | yes | yes |
| 3 | What GPUs was the Transformer trained on? | yes | yes | yes |
| 4 | How many warmup steps were used for the learning rate schedule? | yes | yes | yes |
| 5 | What is the size of the inner layer of the feed-forward network? | yes | yes | yes |
| 6 | What beam size was used when decoding translations? | yes | yes | yes |
| 7 | How large is the shared vocabulary for English-German translation? | yes | yes | yes |
| 8 | How does the model know the order of tokens without recurrence or convolution? | yes | **no** | yes |
| 9 | Which dataset was used for the English constituency parsing experiments? | yes | yes | yes |
| 10 | What does HNSW stand for? | yes | yes | yes |
| 11 | How many simulated characters live in the Generative Agents sandbox? | yes | yes | yes |
| 12 | What limits how much history and instructions an agent can keep in its prompt? | yes | yes | yes |
| 13 | How many tools does ChemCrow give the LLM? | yes | yes | yes |
| 14 | How many API tools are in the API-Bank benchmark? | yes | yes | yes |
| 15 | Which approach hands long-horizon planning to an external classical planner? | yes | yes | yes |
| 16 | What is LSH in approximate nearest neighbor search? | yes | **no** | yes |
| 17 | How many diffusion models make up Imagen Video? | yes | yes | yes |
| 18 | Which architecture does Lumiere use to generate the whole video in one pass? | yes | yes | yes |
| 19 | What kind of architecture does Sora use? | yes | yes | yes |
| 20 | Which method generates video zero-shot without any training? | yes | yes | yes |

## Answers and the judge's reasons

**1. How many parallel attention heads does the Transformer use?**

- Reference: 8 parallel attention heads.
- Answer: The Transformer uses **h = 8** parallel attention heads (layers). [1]
- Correct: yes — The answer correctly states the number of parallel attention heads is 8.
- Grounded: yes — Passage [1] explicitly states "we employ h = 8 parallel attention layers, or heads."

**2. How long did it take to train the base Transformer model?**

- Reference: About 12 hours (100,000 steps).
- Answer: The base Transformer model was trained for a total of 100,000 steps, which took 12 hours [1].
- Correct: yes — The answer correctly states 12 hours and 100,000 steps, matching the reference.
- Grounded: yes — The passage explicitly states the base models were trained for 100,000 steps or 12 hours.

**3. What GPUs was the Transformer trained on?**

- Reference: 8 NVIDIA P100 GPUs, on one machine.
- Answer: The Transformer was trained on **eight P100 GPUs** [1][2][3].
- Correct: yes — The answer correctly identifies the eight P100 GPUs, matching the core fact.
- Grounded: yes — Passages explicitly state training on eight P100 GPUs.

**4. How many warmup steps were used for the learning rate schedule?**

- Reference: 4000 warmup steps.
- Answer: The learning rate schedule used **4,000 warmup steps** [1].
- Correct: yes — The answer correctly states 4,000 warmup steps, matching the reference.
- Grounded: yes — Passage 1 explicitly states "We used warmup_steps = 4000."

**5. What is the size of the inner layer of the feed-forward network?**

- Reference: 2048.
- Answer: The inner layer of the position-wise feed-forward network has a dimensionality of **d_ff = 2048** [1].
- Correct: yes — The answer correctly states the size is 2048, matching the reference.
- Grounded: yes — Passage [1] explicitly states the inner-layer has dimensionality d_ff = 2048.

**6. What beam size was used when decoding translations?**

- Reference: A beam size of 4.
- Answer: A beam size of 4 was used during decoding (beam search) [1].
- Correct: yes — The answer correctly states the beam size is 4, matching the reference.
- Grounded: yes — Passage [1] explicitly states "We used beam search with a beam size of 4".

**7. How large is the shared vocabulary for English-German translation?**

- Reference: About 37,000 tokens.
- Answer: The shared source-target vocabulary for the English-German translation task is about 37,000 tokens [1].
- Correct: yes — The answer correctly states the vocabulary size is about 37,000 tokens.
- Grounded: yes — Passage [1] explicitly states the shared source-target vocabulary is about 37000 tokens.

**8. How does the model know the order of tokens without recurrence or convolution?**

- Reference: Positional encodings, made from sine and cosine functions of different frequencies, are added to the input embeddings.
- Answer: The model knows the order of tokens by adding **positional encodings** to the input embeddings at the bottom of the encoder and decoder stacks [1]. This is necessary because the model contains no recurrence and no convolution, so without these encodings it would have no information about the relative or absolute position of the tokens in the sequence [1]. More broadly, this is needed because the standard self-attention mechanism is permutation-symmetric (permutation-invariant) by default, meaning it treats all tokens symmetrically over all positions—effectively treating the input as a "bag of tokens" rather than an ordered sequence. Injecting positional information (via positional encodings or learned positional embeddings) allows token order to affect the output [Wikipedia: Positional encoding / Transformer].
- Correct: yes — It correctly identifies positional encodings added to input embeddings as the mechanism.
- Grounded: **no** — The claim that self-attention is permutation-symmetric is not stated in the passages.

**9. Which dataset was used for the English constituency parsing experiments?**

- Reference: The Wall Street Journal (WSJ) portion of the Penn Treebank.
- Answer: The English constituency parsing experiments used the Wall Street Journal (WSJ) portion of the Penn Treebank, which contained about 40K training sentences [1]. For the semi-supervised setting, the larger high-confidence and BerkeleyParser corpora (approximately 17M sentences) were also used [1]. Results were evaluated on Section 23 of WSJ [2].
- Correct: yes — The answer correctly identifies the WSJ portion of the Penn Treebank.
- Grounded: yes — All claims about datasets, sentence counts, and evaluation section are directly supported by the text.

**10. What does HNSW stand for?**

- Reference: Hierarchical Navigable Small World.
- Answer: HNSW stands for **Hierarchical Navigable Small World** [1].
- Correct: yes — The answer correctly states the full name matching the reference.
- Grounded: yes — The passage explicitly states HNSW stands for Hierarchical Navigable Small World.

**11. How many simulated characters live in the Generative Agents sandbox?**

- Reference: 25.
- Answer: 25 simulated characters live in the Generative Agents sandbox [1].
- Correct: yes — The answer correctly states the number 25, matching the reference.
- Grounded: yes — Passage [1] explicitly states 25 virtual characters live in the sandbox environment.

**12. What limits how much history and instructions an agent can keep in its prompt?**

- Reference: The finite context length (limited context window).
- Answer: The finite context length (restricted context capacity) limits how much historical information, detailed instructions, API call context, and responses an agent can include in its prompt [1].
- Correct: yes — The answer correctly identifies finite context length as the limiting factor.
- Grounded: yes — The passage explicitly states restricted context capacity limits historical information, instructions, API context, and responses.

**13. How many tools does ChemCrow give the LLM?**

- Reference: 13 expert-designed tools.
- Answer: ChemCrow augments the LLM with 13 expert-designed tools [1].
- Correct: yes — The answer correctly states the number 13 and the nature of the tools.
- Grounded: yes — Passage [1] explicitly states ChemCrow is augmented with 13 expert-designed tools.

**14. How many API tools are in the API-Bank benchmark?**

- Reference: 53 API tools.
- Answer: The API-Bank benchmark contains **53 commonly used API tools** [1].
- Correct: yes — The answer correctly states the number 53, matching the reference fact.
- Grounded: yes — Passage [1] explicitly states it contains 53 commonly used API tools.

**15. Which approach hands long-horizon planning to an external classical planner?**

- Reference: LLM+P.
- Answer: The LLM+P approach (Liu et al. 2023) hands long-horizon planning to an external classical planner. It uses PDDL as an intermediate interface: the LLM translates the problem into “Problem PDDL,” requests a classical planner to generate a PDDL plan from an existing “Domain PDDL,” and then translates the resulting plan back into natural language [1].
- Correct: yes — The answer explicitly identifies LLM+P as the approach using an external classical planner.
- Grounded: yes — All claims match passage [1] exactly regarding LLM+P, PDDL, and steps.

**16. What is LSH in approximate nearest neighbor search?**

- Reference: Locality-Sensitive Hashing: a hashing function maps similar items to the same buckets with high probability.
- Answer: LSH (Locality-Sensitive Hashing) is an approximate nearest neighbor (ANN) technique that uses a hashing function designed so that similar input items are mapped to the same buckets with high probability. The number of buckets is much smaller than the number of inputs, which enables faster retrieval by reducing the search space [1].
- Correct: yes — The answer correctly identifies LSH as Locality-Sensitive Hashing mapping similar items to same buckets.
- Grounded: **no** — "enables faster retrieval" is an inference not explicitly stated in the passage.

**17. How many diffusion models make up Imagen Video?**

- Reference: 7 diffusion models.
- Answer: Imagen Video is made up of **7 diffusion models** in total. [1] These consist of: - 1 base video diffusion model - 3 Temporal Super-Resolution (TSR) models - 3 Spatial Super-Resolution (SSR) models [1]
- Correct: yes — The answer correctly states 7 diffusion models, matching the reference.
- Grounded: yes — The passages explicitly state there are 7 models: 1 base, 3 TSR, and 3 SSR.

**18. Which architecture does Lumiere use to generate the whole video in one pass?**

- Reference: A Space-Time U-Net (STUNet).
- Answer: Lumiere uses a **Space-Time U-Net (STUNet)** architecture to generate the entire temporal duration of the video in a single pass [1][2]. Key details: - STUNet inflates a pretrained text-to-image U-Net so it can downsample and upsample videos in both time and space dimensions, allowing the whole video to be generated at once and removing the need for temporal super-resolution (TSR) components [1][2]. - Its convolution-based blocks consist of pretrained text-to-image layers followed by a factorized space-time convolution, while attention-based blocks at the coarsest U-Net level add temporal attention after the pretrained layers [2]. - Only the newly added layers are further trained [2].
- Correct: yes — The answer correctly identifies Space-Time U-Net (STUNet) as the architecture.
- Grounded: yes — All claims are directly supported by the provided passages.

**19. What kind of architecture does Sora use?**

- Reference: DiT, a Diffusion Transformer, operating on spacetime patches.
- Answer: Sora uses a **Diffusion Transformer (DiT)** architecture. It operates on spacetime patches of video and image latent codes, representing visual input as a sequence of spacetime patches that serve as Transformer input tokens [1]. This is consistent with the broader note that Sora leverages the Transformer architecture for diffusion video modeling [2].
- Correct: yes — It correctly identifies DiT and spacetime patches, matching the reference.
- Grounded: yes — All claims about Sora's DiT architecture, spacetime patches, and Transformer usage are directly supported.

**20. Which method generates video zero-shot without any training?**

- Reference: Text2Video-Zero.
- Answer: Text2Video-Zero (Khachatryan et al. 2023) enables zero-shot, training-free video generation by enhancing a pre-trained image diffusion model with mechanisms for temporal consistency [1].
- Correct: yes — The answer correctly identifies Text2Video-Zero as the method.
- Grounded: yes — The passage explicitly states Text2Video-Zero enables zero-shot, training-free video generation with temporal consistency mechanisms.
