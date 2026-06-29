# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Azure ML Training

This repo includes an Azure ML submission helper for running the reconstructed
Transformer training script on H100:

- [`training/aml/submit_transformer_train.py`](training/aml/submit_transformer_train.py)
- [`training/aml/tinystories_h100_basic.yaml`](training/aml/tinystories_h100_basic.yaml)

The submit script launches [`cs336_basics/transformer_train.py`](cs336_basics/transformer_train.py)
on Azure ML and maps AML datastore inputs to the script's current CLI arguments.

Before submitting, upload the raw `int32` token binary files produced by the
token preprocessing workflow to the ADLS Gen2 paths configured in
[`training/aml/tinystories_h100_basic.yaml`](training/aml/tinystories_h100_basic.yaml):

- `training/TinyStoriesV2-GPT4-train_tokens.bin`
- `training/TinyStoriesV2-GPT4-valid_tokens.bin`

Important: `transformer_train.py` currently reads token data with
`np.memmap(path, dtype=np.int32, mode="r")`, so the AML inputs should be raw
`.bin` files, not `.npy` arrays, unless the training loader is changed.

The basic H100 YAML is configured to avoid AML datastore input mounts because
AISC/Singularity currently reports `NoIdentityOnCompute` when streaming from
the identity-based `poimatcher` datastore. Instead, the submit script stages the
TinyStories raw token binaries from `training/` into the AML code snapshot and
passes local paths such as `data/TinyStoriesV2-GPT4-train_tokens.bin` to the
training script. This is bulkier to upload, but it avoids the failing
`data-capability.UriMountSession` path.

The AML config also enables `job.metrics_backend: "azureml"`, which passes
`--metrics_backend azureml` to `transformer_train.py`. Local runs default to
`--metrics_backend none`, so they continue to print to the terminal without
creating AML metrics. In Azure ML, the script logs training loss, validation
loss/perplexity, learning rate, elapsed time, and checkpoint iterations to the
run metrics pane through the native Azure ML run context.

The experiment plan and report template for learning-rate tuning, batch-size
variation, and generation are in
[`docs/aml_experiment_plan.md`](docs/aml_experiment_plan.md).

If your workspace has a credentialed datastore that works on AISC, you can switch
the YAML back to datastore inputs by setting `data.source: "datastore"`, adding
the datastore paths, and using an AML output mount.

From the repository root, submit the job with:

```powershell
uv run --with azure-ai-ml --with azure-identity --with pyyaml python training\aml\submit_transformer_train.py --config training\aml\tinystories_h100_basic.yaml
```

Update the YAML if your Azure ML workspace, Singularity virtual cluster,
datastore, or token file paths differ.

