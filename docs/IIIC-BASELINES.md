# IIIC author-native scratch adaptations

All four cells use seed0, the same accepted 73,637/15,778/15,780 train/validation/test cohort, sixteen bipolar channels, six labels0–5 and full10-s/2,000-sample windows. The accepted cache stores microvolts/100. No window cropping, temporal compression or replacement test cohort is used. These are explicitly declared new downstream tasks, not claimed reproductions of published IIIC commands.

| Baseline | Native implementation and input adaptation | Fixed recipe |
|---|---|---|
| BIOT | Original standalone multiclass trainer; original `EEGSupervisedPretrainLoader.iiic_load` applied to restored physicalµV and one-hot labels; retain native Q95 and remove only the returned dataset-id flag | Scratch16-channel capacity; batch128,Adam1e-3/wd1e-5,100epochs,patience5,32workers |
| CBraMod | Original TUEV multiclass main/trainer/model; retain storedµV/100 and reshape into ten patches. Only the first classifier input width changes16000→32000; hidden widths1000/200 and six outputs remain native | Batch64,AdamW1e-4/wd.05,50epochs,multi-LR,clip1,smoothing.1,native cosine |
| LaBraM | Original six-class model/main/engine; restore physicalµV before native /100; scratch spatial slots carry no pretrained anatomical claim | Published TUAB budget50epochs,batch64×8accumulation,AdamW5e-4/wd.05,warmup5,scratchlayer_decay1 |
| EEGPT | Original TUAB10-s decoder/main/engine with16→20 input convolution and six-class head; restoreµV before native /100; random trainable encoder | Two ranks,batch100/rank,50epochs,AdamW5e-4/wd.05,warmup5,layer_decay.65;explicit BF16 compatibility runtime |

Configs live under `configs/baselines/<model>/iiic-scratch.*`. Obsolete blocked scratch placeholders are removed when promoted. Original author checkouts and commits remain recorded in `upstream/sources.json`; no generic baseline trainer is substituted.

GPU smokes use only training/validation data, verify finite updates and checkpoint reload, and gate formal jobs through Slurm `afterok`. BIOT/CBraMod/EEGPT use the common source/config/data smoke gate; LaBraM verifies its native `smoke_pass.json` identity. Held-out test evaluation follows checkpoint selection. BIOT's literal released terminal selection and independently archived validation-best checkpoint remain distinct; its validation-selected audit follows the same rule as the other BIOT table cells.

Use the actual native environments. BIOT retains Python3.9/Torch1.13/Lightning1.6.4 and an isolated cuFFT11.8 preload for modern GPUs. LaBraM uses its isolated original-dependency environment. Source freezes, environment receipts, Slurm IDs, outputs and checkpoints live outside the active repository.
