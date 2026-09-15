# Full twelve-dataset Quad16 seed coverage and next-stage decision

## Authorized seed execution
Target: all twelve paper datasets, seeds0/1/2 (36 runs). No new host-swap or pretraining jobs started in this turn.
- TUEV and ADFD: all three complete, do not duplicate.
- CHBMIT: seed0 complete; seeds1/2 launch automatically on145 after current transfer and full checksum match.
- SleepEDF: seed0 complete; seeds1/2 training145 GPU0/1.
- TUAR/Siena: seed0 complete; seeds1/2 have four145 waiters, transfer worker active.
- CAUEEG/ISRUC: six145 waiters (0/1/2), queued transfer after Siena. Require data SHA match and >=10GiB disk margin for new copies. Current GPU6 belongs to another user, must not touch.
- AMD420546: TUSZ0/1/2 + TUEP0, four cards.
- AMD420547: IIIC0/1/2 + TUEP1, four cards.
- AMD420548: TUAB0/1/2 + TUEP2, four cards.
All three AMD allocations running at admission, structure and real-data smoke precede each training. Each node24h, aggregate maximum72 node-hours; release when its four processes finish. Submission filter reported1849.06/2250 used. This is not a guarantee all training schedules fit. Time-budget results must be marked censored.
B2 TUSZ46027491 was confirmedPENDING then cancelled before AMD migration. No duplicated TUSZseed0.
mi2508x test-only rejected24h (account cap12h), so selectedmi2104x24h; no failed paid8card allocation.

New six dataset configs preserve dataset-specific A128 loss, sampling, labels, validation interval/subsampling, sequence flattening and positional encoding; use the same16band Quad architecture and n5 regularization as the fixed breadth candidate. Newdatasetcaps22h; existingTUSZ12h retained across all seeds. Checkpoints retained by defaultRunControl=True.

## Evidence-aware low-budget host plan (not launched)
TFM arxiv2502.16060v3 section4.3/C.4 uses two hosts BIOT and LaBraM, with single/multiple-dataset pretraining; LaBraM's neural tokenizer supplies masked-modeling targets, not just an inference input projection. Our cheaper experiment must state its different scope.
Existing legacyCFM+CBraMod TUEV3seeds improves roughly.564->.612kappa; IIIC decreases roughly.394->.310. These are old8band/patch200 adapter results, notQuad16. CBraMod adapter currently hardcodes8bands, nativewidth200/patch200 and band mean pooling: cannot relabel its config asQuad. Mean pooling is potentially lossy; earlier source commentary asserting preservation does not establish that.
Start two hostsCBraMod andBIOT, three datasetsTUEV/CHB/IIIC. Compare native tokenizer versusQuad16 front-end under same host, random backbone initialization, token count, training recipe and compute reporting. Use deterministic shape adaptation and a minimal learned projection; keep the16bandPAC quadrature construction, document any host-required patch-grid aggregation. Do not inflate token count16x or introduce a large adapter that confounds the comparison. Audit gradient flow, shape, initialization and native control parity before submission. Firstseed0 paired screen (12runs; reuse exact historical controls only after fullprotocol/source audit), then replicate positive/ambiguous evidence rather than claim single-seed superiority. This tests architecture portability without large-scale host pretraining; it does not establish zero-shot compatibility with released pretrained encoders.
If budget permits later, add one released-checkpoint host with matched frozen-backbone/adaptor-only native-control training. Input-distribution shift is a real additional hypothesis; this is secondary, not mandatory to complete the first claim. No claim of a frozen cross-dataset universal tokenizer from independently supervised task-specific frontends.

## Pretraining hypothesis and stop rule (not launched)
No confidence claim of guaranteed benefit. History already covers amplitude, band-normalized/PAC reconstruction, full loading and matched patch sizes, longer schedules and fine-tuning recipes. Quad supervised performance does not prove it benefits fromSSL.
SingleQuad candidate; maskedlatentprediction EMA teacher vs student is a different target hypothesis, not new PAC novelty. Do not repeat reconstruction variants or simultaneously pretrainA128. Raw-input masks precede all nonlocal filtering/Hilbert/PAC computation; audit receptive-field leakage and teacher collapse/variance. Train once on existing eligible pretrain pool, fixed small compute budget with prespecified checkpoints. Compare against matchedQuad scratch using TUEV andCHB validation with sameFT protocol; frozen probes are a cheap diagnostic, not the final success criterion. Before spending, fix pilotbudget, seeds, FT protocol and tolerances from scratch validation variance. Require consistent downstream validation benefit and no material regression on the other keytask before expanding; otherwise stopbudgetescalation. An underpowered short pilot cannot establish that allSSLfails. Test sets must not select objective, checkpoint orhyperparameters.
OldF192/A128 pretrains are historicaldiagnostics, not matchedQuadcontrols. A matchedrawSSL control is necessary if claiming latenttarget specifically causes benefit, but can be deferred until latent vs scratch shows any benefit; disclose missingcontrol meanwhile.
