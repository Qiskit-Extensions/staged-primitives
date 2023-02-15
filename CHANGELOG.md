## 0.4.0 (2023-02-15)

### Feat

- **reckoning**: add support for counts dict inputs
- **reckoning**: add support for pauli str inputs
- **reckoning**: add support for pauli phases

### Refactor

- **reckoning**: auto cast output to python core numeric types

### Perf

- **reckoning**: improve complex abs square computation runtime

## 0.3.1 (2023-02-09)

### Fix

- **reckoning**: compute real valued std_error instead of complex

## 0.3.0 (2023-02-08)

### Feat

- **reckoning**: add support for non-hermitian operators

### Fix

- copy metadata to avoid backend.run update by reference bug in Aer

## 0.2.0 (2023-02-07)

### Feat

- **time**: add elapsed time utils
- **reckoning**: reckon observable, pauli and counts in ExpvalReckoner

### Refactor

- **reckoning**: clean loop in canonical _reckon method

## 0.1.0 (2023-02-06)

### Feat

- **utils**: add time module and isotimestamp function
- **binary**: add binary_digit util

### Refactor

- **operators**: rename operators module to paulis
- **results**: rename results module to counts
- **binary**: change return value in parity_bit to bool

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
