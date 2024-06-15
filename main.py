# 
# Install Packages
# 
!pip install transformers dataset rouge-score datasets peft
!pip install -i https://pypi.org/simple/ bitsandbytes
!pip install huggingface_hub


# 
# OpenELM Settings
# 
"""Module to generate OpenELM output given a model and an input prompt."""
import os
import logging
import time
import argparse
from typing import Optional, Union
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq

from datasets import load_dataset, Dataset

# HF_TOKEN access token
from google.colab import userdata
userdata.get('HF_TOKEN')

# The following function is revised from https://huggingface.co/apple/OpenELM/blob/main/generate_openelm.py
def generate(
    prompt: str,
    model: Union[str, AutoModelForCausalLM],
    hf_access_token: str = None,
    tokenizer: Union[str, AutoTokenizer] = 'meta-llama/Llama-2-7b-hf',
    device: Optional[str] = None,
    max_length: int = 1024,
    assistant_model: Optional[Union[str, AutoModelForCausalLM]] = None,
    generate_kwargs: Optional[dict] = None,
) -> str:
    """ Generates output given a prompt.
    Args:
        prompt: The string prompt.
        model: The LLM Model. If a string is passed, it should be the path to
            the hf converted checkpoint.
        hf_access_token: Hugging face access token.
        tokenizer: Tokenizer instance. If model is set as a string path,
            the tokenizer will be loaded from the checkpoint.
        device: String representation of device to run the model on. If None
            and cuda available it would be set to cuda:0 else cpu.
        max_length: Maximum length of tokens, input prompt + generated tokens.
        assistant_model: If set, this model will be used for
            speculative generation. If a string is passed, it should be the
            path to the hf converted checkpoint.
        generate_kwargs: Extra kwargs passed to the hf generate function.
    Returns:
        output_text: output generated as a string.
        generation_time: generation time in seconds.
    Raises:
        ValueError: If device is set to CUDA but no CUDA device is detected.
        ValueError: If tokenizer is not set.
        ValueError: If hf_access_token is not specified.
    """
    if not device:
        if torch.cuda.is_available() and torch.cuda.device_count():
            device = "cuda:0"
            logging.warning(
                'inference device is not set, using cuda:0, %s',
                torch.cuda.get_device_name(0)
            )
        else:
            device = 'cpu'
            logging.warning(
                (
                    'No CUDA device detected, using cpu, '
                    'expect slower speeds.'
                )
            )

    if 'cuda' in device and not torch.cuda.is_available():
        raise ValueError('CUDA device requested but no CUDA device detected.')

    if not tokenizer:
        raise ValueError('Tokenizer is not set in the generate function.')

    if not hf_access_token:
        raise ValueError((
            'Hugging face access token needs to be specified. '
            'Please refer to https://huggingface.co/docs/hub/security-tokens'
            ' to obtain one.'
            )
        )

    if isinstance(model, str):
        checkpoint_path = model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
    model.to(device).eval()
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            token=hf_access_token,
        )

    # Speculative mode
    draft_model = None
    if assistant_model:
        draft_model = assistant_model
        if isinstance(assistant_model, str):
            draft_model = AutoModelForCausalLM.from_pretrained(
                assistant_model,
                trust_remote_code=True
            )
        draft_model.to(device).eval()

    # Prepare the prompt
    tokenized_prompt = tokenizer(prompt)
    tokenized_prompt = torch.tensor(
        tokenized_prompt['input_ids'],
        device=device
    )

    tokenized_prompt = tokenized_prompt.unsqueeze(0)


    # Generate
    stime = time.time()
    output_ids = model.generate(
        tokenized_prompt,
        max_length=max_length,
        pad_token_id=0,
        assistant_model=draft_model,
        **(generate_kwargs if generate_kwargs else {}),
    )
    generation_time = time.time() - stime

    output_text = tokenizer.decode(
        output_ids[0][tokenized_prompt.shape[1]:].tolist(),
        skip_special_tokens=True
    )

    return output_text, generation_time

def extract_sentence(abstract: str, model) -> str:
  try:
    """
    prompt = (
        "You will receive an abstract paragraph from me."
        "Please extract the part describing the methodology from the abstract"
        "Return content only and don't be formulatic, no break line and other irrelevant information.\n\n"
        "Abstract:\n%s\n\n"
    ) % abstract
    """
    prompt = "From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information."
    input = f"<s>[INST]<<SYS>>{prompt}<</SYS>>{abstract}[/INST]"
    output_text, genertaion_time = generate(
        prompt=input,
        model=model,
        hf_access_token=userdata.get('HF_TOKEN')
    )
    return output_text
  except Exception as e:
    print(f"Exception occurred: {e}")
    return None

# Llama2 tokenizer have the specific format of the prompt, and the prompt and the abstract should be gathered to the input
# Use -100 to occupy len(instruction) before the response, otherwise, the instruction and labels will overlapped
def preprocess_function(example):
    prompt = "From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information."
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<s>[INST]<<SYS>>{prompt}<</SYS>>{example['abstract']}[/INST]", add_special_tokens=False)
    response = tokenizer(f"{example['methodology']}</s>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

model_name = "apple/OpenELM-270M-Instruct"
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=userdata.get('HF_TOKEN'))

# 
# QLoRA
# 
# use BitsAndBytesConfig to do quantinization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)
lora_model = get_peft_model(model, lora_config)


# 
# Feed abstract to Llama3 to get the methodology and write the tuple into data.csv
# 
dataset = load_dataset("ICLR2024/ICLR2024-papers")

# how many rows will be generated
collected_size = 200

from transformers import pipeline, AutoModelForQuestionAnswering
import csv

pipeline = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}, # load_in_4bit: quantize
    device_map="auto",
)

abstracts = []
methodologies = []
start_index = 800
for i, abstract in enumerate(dataset['train']['abstract'][start_index:],start=start_index):
    if i == start_index + collected_size:
        break
    messages = [
      {"role": "system", "content": "You will receive an abstract paragraph from me, please describe the methodology from the abstract in a short paragraph."},
      {"role": "user", "content": abstract},
    ]
    terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
      messages,
      max_new_tokens=512,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
    )
    methodology = outputs[0]['generated_text'][2]['content']
    methodologies.append(methodology)
    abstracts.append(abstract)
    print("\nabstract: ")
    print(abstract)
    print("\nmethodology: ")
    print(methodology)

with open('dataset2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['abstract', 'methodology'])
    for abstract, methodology in zip(abstracts, methodologies):
        writer.writerow([abstract, methodology])


# 
# Test the rouge-score for the methodology output from Llama3
# 
abstract = "The reliability of self-labeled data is an important issue when the data are regarded as ground-truth for training and testing learning-based models.This paper addresses the issue of false-alarm hashtags in the self-labeled data for irony detection.We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.Furthermore, we apply our model to prune the self-labeled training data.Experimental results show that the irony detection model trained on the less but cleaner training instances outperforms the models trained on all data."
messages = [
  {"role": "system", "content": "You will receive an abstract paragraph from me, please describe the methodology from the abstract in a short paragraph."},
  {"role": "user", "content": abstract},
]
terminators = [
  pipeline.tokenizer.eos_token_id,
  pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = pipeline(
  messages,
  max_new_tokens=512,
  eos_token_id=terminators,
  do_sample=True,
  temperature=0.6,
  top_p=0.9,
)
methodology = outputs[0]['generated_text'][2]['content']
methodologies.append(methodology)
abstracts.append(abstract)
print("\nabstract: ")
print(abstract)
print("\nmethodology: ")
print(methodology)

# 
# Train OpenELM-270M-Instruct
# 

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('input4.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# format train_data
train_dataset = train_dataset.map(preprocess_function)
eval_dataset = eval_dataset.map(preprocess_function)

# training arguments
training_args = TrainingArguments(
    output_dir="./NLP-2024",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    eval_strategy="epoch",
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()


# 
# Give the abstract and get the predicted abstract
# 
tested_abstract = "The reliability of self-labeled data is an important issue when the data are regarded as ground-truth for training and testing learning-based models.This paper addresses the issue of false-alarm hashtags in the self-labeled data for irony detection.We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.Furthermore, we apply our model to prune the self-labeled training data.Experimental results show that the irony detection model trained on the less but cleaner training instances outperforms the models trained on all data."
tested_methodology = extract_sentence(tested_abstract, lora_model)
print("Predicted Methodology: ")
print(tested_methodology)
print("\n")


# 
# Push Model to huggingface
# 
trainer.save_model("./NLP-2024")
tokenizer.save_pretrained("./NLP-2024")

from huggingface_hub import notebook_login, HfApi, HfFolder
notebook_login()
folder = HfFolder()
token = folder.get_token()
api = HfApi()
repo_url = api.create_repo(token, 'model_name', organization="royshao", private=False, repo_type="model")
!git -C ./NLP-2024 push {repo_url}


# 
# Load the fine-tuned model from hugging face and evaluate
# 
custom_model = AutoModelForCausalLM.from_pretrained(
    "royshao/NLP-2024",
    trust_remote_code=True
)

def evaluate(foo, model):
    import urllib.request
    test = "https://www.cs.nccu.edu.tw/~hhhuang/courses/nlp2024/test2024.in"
    gold = "https://www.cs.nccu.edu.tw/~hhhuang/courses/nlp2024/test2024.gold"

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'])

    total = 0
    cnt = 0
    with urllib.request.urlopen(test) as testin, \
         urllib.request.urlopen(gold) as gold:
        for input, ref in zip(testin, gold):
            input = input.decode("utf-8")
            ref = ref.decode("utf-8")
            output = foo(input, model)
            score = scorer.score(ref, output)['rougeL'].fmeasure
            cnt += 1
            total += score
            print("Test case %d: %g" % (cnt, score))
    print("Overall: %g" % (total / cnt))
    return total / cnt

# As your working function is `extract_sentence`, so do evaluation with the following statement
evaluate(extract_sentence, custom_model)
