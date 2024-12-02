DP-OPT: MRP Final Project
====================================================

## Overview


Large Language Models (LLMs) have emerged as dominant tools for various tasks, particularly when tailored for a specific target by prompt tuning. Nevertheless, concerns surrounding data privacy present obstacles due to the tuned prompts' dependency on sensitive private information. A practical solution is to host a local LLM and optimize a soft prompt privately using data. Yet, hosting a local model becomes problematic when model ownership is protected. Alternative methods, like sending data to the model’s provider for training, intensify these privacy issues facing an untrusted provider. In this paper, we present a novel solution called Differentially-Private Offsite Prompt Tuning (DP-OPT) to address this challenge. Our approach involves tuning a discrete prompt on the client side and then applying it to the desired cloud models. We demonstrate that prompts suggested by LLMs themselves can be transferred without compromising performance significantly. To ensure that the prompts do not leak private information, we introduce the first private prompt generation mechanism, by a differentially-private (DP) ensemble of in-context learning with private demonstrations. With DP-OPT, generating privacy-preserving prompts by Vicuna-7b can yield competitive performance compared to non-private in-context learning on GPT3.5 or local private prompt tuning.

## Get Started

Prepare conda env.
```shell
conda create --name dp-opt python=3.8 -y
conda activate dp-opt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate sentencepiece scikit-learn wandb autodp
# transformers==4.28.1
```

Prepare DLN datasets
```shell
bash setup_data.sh
```

> :warning: **Warning:** Setting `echo` and `logprobs` simultaneously is no longer supported for certain OpenAI models.
> However, classification inference with openai models requires both settings. Consider to host your own models, e.g., thru vLLM, instead.

**Example**: Do prompt engineer on website:
```shell
pip install gradio
python web_demo.py
# open http://127.0.0.1:7860
```
**Example**: Use local model (`lmsys/vicuna-7b-v1.3`) to generate a instruction and test the instruction by OpenAI model (`text-davinci-003`).
* OPT:
```shell
# generate a instruction
python train_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 --holdout_ratio=0.01
# evaluate the instruction
python eval_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 \
--test_model=text-davinci-003
```
* DP-OPT:
```shell
# generate a instruction
python train_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 --holdout_ratio=0.01 \
--target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True
# evaluate the instruction
python eval_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 \
--target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True \
--test_model=text-davinci-003
```

## Experiments

Most of the codes are based off of [DP-OPT]((https://github.com/VITA-Group/DP-OPT)).

