## Alignment With Current Pipeline

* Training remains at image/bag level: 12 patches per image aggregated via attention; no pile-level training.

* Evaluation stays pile-level by mean pooling of image-level predictions across images in a pile.

* All adaptation losses (DANN, MMD, Orthogonal) operate on bag-level features and predictions.

## Approach Overview

* Shared backbone + attention aggregator + classifier head for both domains.

* Add a domain discriminator fed by bag-level feature `z` with Gradient Reversal Layer (GRL) for adversarial domain confusion.

* Align distributions via multi-kernel MMD on `z` (class-conditional preferred since target is labeled).

* Reduce interference via orthogonal regularization between task and domain heads; preserve class boundaries.

* Optimize: `L_total = L_cls + λ_adv·L_adv + λ_mmd·L_mmd + λ_orth·L_orth`.

## Model Architecture Changes

* Use existing MIL classifier `classification_model/src/models/bma_mil_model.py:130-199`; expose bag feature `z` from `AttentionAggregator` before classifier.

* Add `DomainDiscriminator(z)` (MLP with spectral norm): `Linear(d→d/2)→ReLU→Dropout→Linear(d/2→1)` with sigmoid.

* Insert `GradientReversal(λ_grl)` before `DomainDiscriminator`.

* Keep one shared classifier head for both domains.

## Losses (Bag/Image Level)

* Classification (both domains):

  * `L_cls_source = CrossEntropy(qld1_logits, qld1_labels)`

  * `L_cls_target = CrossEntropy(qld2_logits, qld2_labels)`

  * `L_cls = L_cls_source + L_cls_target`

* Adversarial (DANN): domain labels `d=0` for QLD1, `d=1` for QLD2.

  * `L_adv = BCE(domain_pred(qld1_z), 0) + BCE(domain_pred(qld2_z), 1)` (GRL active).

* MMD (bag-level `z`, multi-kernel RBF):

  * `L_mmd = Σ_k MMD_k(z_s, z_t)`, `σ_k ∈ {0.5, 1, 2, 4}`.

  * Preferred class-conditional: per-class `MMD(z_s^c, z_t^c)` and sum over `c∈{cat1,cat2,cat3}`.

* Orthogonal regularization:

  * Primary: decouple heads via `L_orth = ||W_cls W_dom^T||_F`.

  * Optional: prototype safety term preventing domain shift along class-separating axes (`μ_s^c, μ_t^c` from batch; penalize `cos(μ_t^c−μ_s^c, μ_s^i−μ_s^j)`).

## Data & Dataloaders (Bag/Image Level Only)

* Create two `BMADataset` instances `classification_model/src/data/dataset.py:12-90` for QLD1 and QLD2.

* Build two bag-level loaders with identical `batch_size`/`collate_fn` (same as current image-level training).

* Iterate jointly per step: draw one mini-batch from each; forward through shared model; compute domain/adaptation losses on bag-level features `z`.

* Balance sampling using cycling of the smaller loader to keep per-epoch exposure similar.

## Training Loop Integration

* Update `train_one_epoch` `classification_model/src/utils/training.py:36-85` to consume dual loaders zipped by iteration.

* Per iteration:

  * Forward QLD1 batch → `(qld1_logits, qld1_z)`; QLD2 batch → `(qld2_logits, qld2_z)`.

  * Compute `L_cls` from both; `L_adv` from GRL+domain discriminator on `z`; `L_mmd` on `z_s` vs `z_t` (optionally per-class masks); `L_orth` from `W_cls,W_dom`.

  * Combine to `L_total`; single backward/step; scheduler behavior unchanged `training.py:328-366`.

* Validation/Evaluation:

  * Keep existing image-level validation for both domains.

  * Pile-level evaluation by mean pooling of image-level predictions across piles remains unchanged; use existing utilities `classification_model/src/utils/evaluation.py:12-159`.

  * Early stopping on QLD2 weighted F1.

## Hyperparameters & Schedules

* New config fields `classification_model/configs/config.py`:

  * `LAMBDA_ADV` (ramp `0→1` over first 5 epochs), `GRL_COEFF` tied to `LAMBDA_ADV`.

  * `LAMBDA_MMD` (e.g., `0.5`, ramp over first 5 epochs), `MMD_BANDWIDTHS = [0.5,1,2,4]`.

  * `LAMBDA_ORTH = 0.01` constant.

  * Toggle `USE_CLASS_COND_MMD = True`.

* Optimizer/scheduler: reuse existing AdamW + ReduceLROnPlateau `training.py:240-265`.

## Stability & Guidelines Compliance

* Operate strictly on bag/image-level features; no pile-level training changes.

* Spectral norm on domain discriminator; label smoothing (`0.05`) for domain labels.

* Ramp-ups for `λ_adv`, `λ_mmd`, `λ_grl` to prevent early collapse.

* Gradient clipping (`max_norm=5.0`).

## Verification & Logging

* Log per-domain metrics and all loss components per epoch.

* MMD self-test: identical inputs ≈ 0; non-identical > 0.

* GRL sign test on a toy layer; monitor `||W_cls W_dom^T||_F`.

* Ablations: baseline, +DANN, +MMD, +DANN+MMD, +DANN+MMD+Orth.

## Integration Steps (Files)

* `classification_model/src/models/domain_discriminator.py`: `GradientReversal`, `DomainDiscriminator`.

* `classification_model/src/losses/mmd.py`: multi-kernel RBF MMD (supports class-conditional masks).

* `classification_model/src/losses/orthogonal.py`: `||W_cls W_dom^T||_F` (+ optional prototype variant).

* `classification_model/src/models/bma_mil_model.py`: expose `z` alongside logits.

* `classification_model/src/utils/training.py`: dual-loader joint training; compute new losses; preserve evaluation.

* `classification_model/configs/config.py`: new hyperparameters.

* `classification_model/scripts/train.py`: construct QLD1/QLD2 loaders and pass to training.

## Initial Hyperparameters

* `λ_adv = 1.0` (ramp 5 epochs), `GRL_COEFF = 1.0`.

* `λ_mmd = 0.5` (ramp 5 epochs), `bandwidths = [0.5,1,2,4]`.

* `λ_orth = 0.01` constant.

## Expected Outcomes

* Target (QLD2) bag/image-level predictions improve via domain invariance and distribution alignment; pile-level metrics improve after mean pooling.

* Source performance is maintained via joint supervised training and orthogonal decoupling of task/domain signals.

