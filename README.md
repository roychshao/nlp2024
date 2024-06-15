# NLP2024
---
### Target 
* fine-tuning a OpenELM based model to extract methodology from an abstract

### Methodology
* We take ICLR2024 papers as abstact dataset and feed it to Llama3 to get the methodology.
* Use this dataset to train OpenELM.
* Use QLoRA to train OpenELM.

### Some notes
* Tokenizer of OpenELM use Llama2
* In the preprocess function, we use Llama2 formatted instruction.
* The instruction should contains both prompt and our abstracts, and labels is corresponding methodologies.
* The overall rouge-score of the whole evaluate dataset is 0.49, some can be 1 and some get pretty low score.
