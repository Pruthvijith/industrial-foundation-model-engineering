# SmolLM2 engineering choice adapter

The measured DPO adapter SHA-256 is 897f48a8d2ad742140bc81f9a18fcca74d808078fa9025e4ecbbb17c94dd6d1a. Rebuild from configs/cpu.json and the pinned base. Its four-choice interface expects the exact mapping supplied in each lab record. Decode the single completion token corresponding to a letter preceded by a space. Candidate output needs an independently verified option map.

Training uses 96 synthetic cases with programmatic preferences. The 48-case holdout shares the underlying domain families with training. Accuracy was 29.2%, and seven of twelve incomplete-record cases selected the intended answer. Free-text arithmetic failed the output contract across all 72 challenges. The model remains held for research use. It must not be used for aircraft, maintenance or safety decisions.

Results, hardware, loss traces, split digests and raw predictions are under reports/. The base model and its licence remain with its publisher. Only source and measured metadata are distributed here.
