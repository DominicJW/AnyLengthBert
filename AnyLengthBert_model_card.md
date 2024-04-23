---
{}
---
language: en
license: Apache 2.0
tags:
- text-classification
- evidence retrieval
repo: https://github.com/DominicJW/AnyLengthBert

---

# Model Card for g15612dj-r21857jl-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether detect whether once sentence may provide evidence for the other. As the name suggests it can accept any lengthed input


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a BERT model and it was fine-tuned
      on 25K pairs of texts.  It is designed to not truncate text, but to accept a tensor of chunks of tokenised text, which each is fed into BERT. The outputs of each chunk are pooled. 
      
      

- **Developed by:** Dominic Johnston-Whiteley
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** bert-base-uncased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

30K pairs of texts provided by the instructor of COMP34812. "[REF]" was removed from evidence sentences, and "We should" removed from claim sentences as almost every claim began with "We should" and it does not really provide any information. This hopefully makes the model more genralisable

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - train_batch_size: 32
      - eval_batch_size: 32
      - seed: 1
      - num_epochs: 2
      -learn_rate_scheduler: linear with warmup
      -warmup_steps: 500
      -Total steps: ~100K (4 epochs), trains for two but scheduler thinks 4 so learn rate only decreases by 1/2 
      -dropout 0.1
      -max length: 128
      

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


       -at max length 120:
      - overall training time: 3 minutes
      - duration per training epoch: 1 minute 30 seconds
      - model size: 454MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the training set provided, amounting to 2.5K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      -Precision: 0.76,
      -Recall: 0.76,
      -F1 Score: 0.76
      -MCC: 0.67,
      -Accuracy: 0.87,
      

### Results

The model obtained an F1-score of 76% and an Accuracy of 87%, a Mathews Correlation Coefficent of 67%, a Recall of 76% and a Precision of 76%.

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 30GB,
      - GPU: L4 used for training, T4 is acceptable

### Software


      - Transformers 4.4.0
      - Pytorch 2.2.1+cu121

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Training data has not been fully vetted by me. But some examples do stick out

    claim: We should legalize same sex marriage 
    evidence: A June 2006 TNS-Sofres poll found that 45% of respondents supported same-sex marriage, with 51% opposed. 
    label: 1
    
    claim: We should increase internet censorship
    evidence: According to the report, few countries demonstrated any gains in Internet freedom, 
            and the improvements that were recorded reflected less vigorous application of existing controls rather than new steps taken by governments to actively increase Internet freedom.
    label:1
    
    these examples were not difficult to find. 
    The name of the task is Evidence Detection. I think the task is really Evidence Retrieval, as it seems these labels are more saying if the evidence pertains to the claim. 
    

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by iterative human chosen trials
      with different values. Start points found from related works credited in readme.
