# Steps GLAD #
Copyright (C) 2015 Naoki Otani
Kyoto University

Please use the following reference for citation of work that benefits from use
of any portion of the Steps GLAD software:

```
@inproceedings{Otani2015Quality,
author = {Otani, Naoki and Baba, Yukino and Kashima, Hisashi},
title = {Quality control for crowdsourced hierarchical classification},
booktitle = {2015 IEEE International Conference on Data Mining (ICDM)},
doi = {10.1109/ICDM.2015.83},
pages = {937--942},
year = {2015}
}
```

For questions and comments, plase contact Naoki Otani at: otani.naoki.65v@st.kyoto-u.ac.jp

## Installation ##
1. Install GSL, if not already installed.
    * Download .tar.gz from the official page
    * `./configure (--prefix={target dir})`
    * `make`
    * `make install`
2. Modify Makefile to point to the locations of the GSL.
3. Run make.
4. You may have to set `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` to point to the GSL and CBLAS libraries.
5. Run the demo `./bin/glad_steps data/data_steps.txt`.


## Input ##

### Labels ###
In the 1st row:
```
    <#labels> <#workers> <#tasks> <#classes> <#steps>
```

followed by
```
    <taskID> <workerID> <labelId>
```
All IDs start from 0.

See `data/sample_labels.dat`.

### Class hierarchies ###
Suppose a class hierarchy has N tiers.
```
<classID at N-th tier> <ClassID at 1st tier> <ClassID at 2nd tier>...<ClassID at (N-1)th tier>
```
All IDs start from 0 for each tier.

See `data/sample_hierarchy.dat`.

## Usage ##
```
  ./bin/steps_glad <input> <hierarchy> [options]
```
See `./bin/steps_glad -h` for details.

### Mode ###
We can specify the model setting by `-m <int>` option.
- `1`: Steps GLAD with a task dependent approach
- `2`: Steps GLAD with a class dependent approach
- `3`: Steps GLAD with a task-and-class dependent approach
- `4`: Steps Rasch model (as an extention example, not presented in the paper)

### Example ###
```
  mkdir sample_output
  ./bin/steps_glad data/sample_labels.dat data/sample_hierarchy.dat -p sample_output/model0
```
The results will be written in `sample_output/`:

- `model0_alpha.csv`: worker accuracy parameters
- `model0_beta.csv`: task difficulty parameters
- `model0_probs.csv`: probabilities


## NOTE ##
This software was implemented based on GLAD, proposed the following paper.

Whitehill, J., Wu, T., Bergsma, J., Movellan, J. R., & Ruvolo, P. L. (2009). Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise. In Advances in Neural Information Processing Systems 22 (NIPS) (pp. 2035–2043).
