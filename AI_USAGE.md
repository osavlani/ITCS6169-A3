# AI Usage

## Tool(s) used
Claude (Anthropic) — used as a coding/research assistant throughout the project: choosing approaches, writing and reviewing the notebook cells, debugging, and drafting the README and report text.

## Representative prompts and how AI assisted

### 1. Listing one-shot approaches and what fits on 8 GB of VRAM
**Prompt:** "so what are the possible approaches"

**Outcome:** AI listed 11 approaches: zero-shot text, prompt ensembling, one-shot image prototype, text + image fusion, learned adapters (CoOp, Tip-Adapter), model ensembling (CLIP, SigLIP, DINOv2), test-time prompt tuning (TPT), transductive refinement, self-training on pseudo-labels, diffusion augmentation of the support image, and SSL on the foundation model. For each it gave the idea, pros and cons and whether it was realistic on an 8 GB laptop GPU. All the frozen-model options fit easily. Learned adapters were expected to overfit with one image per class, and diffusion variants were heavy and a poor fit for 8 GB. This gave me what I needed to pick which approaches to run myself (see the decision section below).

### 2. A warning in the CLIP load cell
**Prompt:** "my notebook prints a QuickGELU mismatch warning when CLIP loads, in two cells, can you check if my results are affected"

**Outcome:** AI traced the warning to the notebook's original implementation. The OpenAI CLIP weights were trained with the QuickGELU activation, but the model was created as `ViT-L-14`, which uses the standard one, so the weights ran with the wrong activation and every CLIP number (zero-shot, prototypes, fusion, TPT) was slightly off while the code still ran without an error. The fix was to create the model as `ViT-L-14-quickgelu` in the CLIP load cell and in the TPT function. The cached CLIP features and TPT logits were stale, so a one-time cell deleted them (the DINOv2 cache stayed valid) and everything from the CLIP load down was re-run. The numbers did move, for example the one-prompt CLIP baseline went from 88.25% to 86.00%, which showed the old baseline had been a lucky single-prompt number, and the prompt ensemble went from 88.75% to 89.00%. I checked the fix by looking at the load line (`loaded clip ViT-L-14-quickgelu openai`) and confirming the warning was gone.

### 3. No validation set: does testing directly affect the accuracy?
**Prompt:** "as there is no validation set, does testing directly on the test set affect the accuracy we report"

**Outcome:** AI explained that choosing weights or methods by looking at test accuracy does not change the model, but it makes the reported number optimistic, because the test set then helps pick the settings. With one labelled image per class there is nothing to hold out, and 400 test images already give a few points of noise, so small gaps are not meaningful. So all settings were fixed in advance (10 prompt templates, temperatures, fusion weight 1.0, 5 refinement rounds), the fusion-weight and round sweeps were printed for information only and not used to pick a value, and the draw of the 16 support images was repeated over 5 seeds to see how much it matters (76.30% ± 4.51% for the image-only prototypes). The best method was still selected on the test set, which is stated in the report and README.

### 4. What to save when nothing is trained
**Prompt:** "as we are not training any model and wont save checkpoints, what exactly do we save, and how do we avoid losing the runs if the pc restarts, the inference results I mean"

**Outcome:** AI pointed out that all encoders are frozen, so there are no weights to checkpoint. The final checkpoint only needs the class text embeddings and the settings, and the encoders are downloaded by name. What can actually be lost is the expensive inference, which is the test features and the TPT logits (TPT takes about 12 minutes). These went into a disk cache (`results/cache/`) keyed by seed, test size, class names and settings, and each method's accuracy is written to `results/results.json` after it runs. I checked it by restarting the kernel and seeing `loaded from cache: test_feats_...` in the output.

### 5. Cross-checking the report and citations
**Prompt:** "this is my report, add the references and citations like last time, cross check the numbers against the notebook and flag anything that concerns you"

**Outcome:** AI added 12 numbered inline citations and a References list (one-shot learning, CLIP, SigLIP, DINOv2, OpenCLIP, CoOp, Tip-Adapter, prototypical and matching networks, TPT, TIM, Laplacian few-shot), each with its arXiv ID so it can be checked directly. It also checked every percentage in the report against the notebook outputs (all matched) and flagged that the conclusion needed to say that the best method was selected on the test set and that the refinement uses unlabelled test images. Only two references (Laplacian regularized few-shot learning and TIM) were checked against their arXiv pages during the session. The other entries came from the AI's memory, so I check each arXiv ID against its listed title before submitting.

## Decision made independently

From the list of possible approaches, I chose to run zero-shot text with prompt ensembling, model ensembling (CLIP, SigLIP, DINOv2), one-shot image prototypes and TPT, with the diffusion variants only if time permitted, instead of taking AI's recommended stack as it was. I also treated the zero-shot CLIP classifier as the baseline and not as an experiment, and consolidated the experiments into five approaches. Because no validation set exists, I chose to say in the report that the best method was selected on the test set and that the refinement uses unlabelled test images.
