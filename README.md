# INTENT: <u>IN</u>teractive <u>TEN</u>sor <u>T</u>ransformation Synthesis
INTENT is a novel interactive program synthesizer for TensorFlow. It enables end users to *interactively* decompose a complex tensor transformation task to simpler ones to be solved by a synthesizer. This codebase contains the official implementation of the tool from our paper:

**[INTENT: Interactive Tensor Transformation Synthesis](https://web.eecs.umich.edu/~xwangsd/pubs/uist22b.pdf)** </br>
*Zhanhui Zhou\*, Man To Tang\*, Qiping Pan\*, Shangyin Tan, Xinyu Wang, Tianyi Zhang* </br>
Symposium on User Interface Software and Technology (UIST) 2022



This implementation is based on [TF-Coder](https://github.com/google-research/tensorflow-coder), which is a program synthesis tool that runs a combinatorial search to find TensorFlow expressions. The `tensorflow-coder` package is modified to support extra features for decomposition and visualization.


## Interface
![image info](./INTENT.png)

## Abstract
There is a growing interest in adopting Deep Learning (DL) given its superior performance in many domains. However, modern DL frameworks such as TensorFlow often come with a steep learning curve. In this work, we propose INTENT, an interactive system that infers user intent and generates corresponding TensorFlow code on behalf of users. INTENT helps users understand and validate the semantics of generated code by rendering individual tensor transformation steps with intermediate results and element-wise data provenance. Users can further guide INTENT by marking certain
TensorFlow operators as desired or undesired, or directly manipulating the generated code. A within-subjects user study with 18 participants shows that users can finish programming tasks in TensorFlow more successfully with only half the time, compared with a variant of INTENT that has no interaction or visualization support.

## Installation

### Package manager
**Miniconda (optional)**
Miniconda is a light-weight open source package management system for python.

Checkout https://docs.conda.io/en/latest/miniconda.html for installation instructions.
```
# create an env for int and keep it activated during development
$ conda create --name intent python=3.9
$ conda activate intent
```
**Yarn (required)**
Yarn is a package management system for js.

Checkout https://classic.yarnpkg.com/lang/en/docs/install/#debian-stable for installation instructions.

### Project dependency
**tensorflow-coder**
```
$ pip install -e ./tensorflow-coder
```
**server**
```
$ pip install -e ./server
```
**client**
```
$ cd client
$ yarn
```
## Launch INTENT
```
$ cd client
$ yarn build
$ cd ..
$ cd server
$ sh bin/INTENT_run.sh
```
Then go to http://localhost:8000 in your browser (chrome recommended).


## Credits
The project was developed under the supervision of [Prof. Tianyi Zhang](https://tianyi-zhang.github.io/) at Purdue and [Prof. Xinyu Wang](https://web.eecs.umich.edu/~xwangsd/) at UMich. Special thanks to [Yitao Huang](https://github.com/Yitao-Huang) and [Lyubing Qiang](https://github.com/EvelynQiang) for their contributions to the first version of INTENT.

## Citation
If you found our paper/code useful in your research, please consider citing:

```
@inproceedings{zhou2022intent,
 author = {Zhou, Zhanhui and Tang, Man To and Pan, Qiping and Tan, Shangyin and Wang, Xinyu and Zhang, Tianyi},
 title = {INTENT: Interactive Tensor Transformation Synthesis},
 booktitle = {Proceedings of the 35nd Annual ACM Symposium on User Interface Software and Technology},
 year = {2022},
} 
```