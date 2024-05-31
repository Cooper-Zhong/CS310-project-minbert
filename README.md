# minBERT and Downstream Multitasking

This is the default project of CS310 Natural Language Processing at SUSTech, 2024 Spring.

The tasks are:
- Implement the `BERT` model and `AdamW` optimizer based on the skeleton
- Train a BERT model for 3 downstream tasks: sentiment classification, paraphrase detection, and semantic text similarity

Check out our report [here](./report.pdf) for more details.

## Implementation

We mainly used the following strategies:

1. Even Batching
2. Loss Rescaling
3. Additional Pretraining
4. Cross Encoding
5. Task Head Dropout

checkout the `bert_classifier` folder for the source code, some explanations:
- `multitask_classifier.py`: Basic implementation from [here](https://github.com/yuanwang98/ywang09-minbert-default-final-project)
- `multitask_classifier_ram.py`: Adapted from [ramvenkat98](https://github.com/ramvenkat98/methods-to-improve-downstream-generalization-of-minbert)
- `multitask_classifier_crossencode`: Use cross encoding in the task heads for paraphrase and STS
- `multitask_classifier_crossencode_parallel.py`: Support parallel training on multiple gpus.
- `./additional_pretraining_data` : For further pretraining

## Result and Analysis

checkout `./results` and `./Analysis`

## Reference

- https://github.com/yuanwang98/ywang09-minbert-default-final-project
- https://github.com/ramvenkat98/methods-to-improve-downstream-generalization-of-minbert