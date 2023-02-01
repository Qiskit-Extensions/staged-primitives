## 0.0.0 (2023-02-02)

### Feat

- **sampler**: add StagedSampler class
- **base**: add BaseStagedSampler
- **estimator**: add StagedEstimator implementation
- **utils.circuits**: add infer_end_layout_intlist
- **utils.results**: add map_counts util
- **utils.results**: add reckoning submodule
- **utils**: add results module
- **utils.operators**: add decomposition submodule
- **utils**: add compose_circuits_w_metadata
- **base**: add BaseStagedEstimator
- **utils**: add operators utils
- **utils**: add circuits utils
- **utils**: add binary utils

### Fix

- **results**: use int_outcomes instead of int_raw

### Refactor

- **sampler**: simplify _transpile_single_unbound logic
- **estimator**: rearrange init
- **utils.results**: rename mask_counts to bitmask_counts
- **utils.results**: rename counts_list arg  in ExpvalReckoner
- **utils.operators**: rename init_observable to normalize_operator
- **base**: remove backend abstractproperty
