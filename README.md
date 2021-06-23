# ✨ MyDummyExecutor

# LaserEncoder

**LaserEncoder** is a encoder based on Facebook Research's LASER (Language-Agnostic SEntence Representations) to compute multilingual sentence embeddings.

It encodes `Document` content from an 1d array of string in size `B` into an ndarray in size `B x D`.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

```bash
python -m laserembeddings download-models
```


## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://MyDummyExecutor')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://MyDummyExecutor'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://MyDummyExecutor')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://MyDummyExecutor'
```


### 📦️ Via Pypi

1. Install the `jinahub-MY-DUMMY-EXECUTOR` package.

	```bash
	pip install git+https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	```

1. Use `jinahub-MY-DUMMY-EXECUTOR` in your code

	```python
	from jina import Flow
	from jinahub.SUB_PACKAGE_NAME.MODULE_NAME import MyDummyExecutor
	
	f = Flow().add(uses=MyDummyExecutor)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	cd EXECUTOR_REPO_NAME
	docker build -t my-dummy-executor-image .
	```

1. Use `my-dummy-executor-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://my-dummy-executor-image:latest')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://MyDummyExecutor')

with f:
    resp = f.post(on='foo', inputs=Document(), return_resutls=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of the shape `256`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.


## 🔍️ Reference
- Some reference

