# AnyLengthBert

The file names are quite self explanatory. Use the AnyLengthBert.ipynb for all of these files in one (this is the notebook I used whilst developing the model)

this notebook was used as a base https://colab.research.google.com/github/NadirEM/nlp-notebooks/blob/master/Fine_tune_ALBERT_sentence_pair_classification.ipynb

this tutorial was used to advise hyper parameters https://medium.com/@behitek/fine-tune-bert-for-the-sentence-pair-classification-task-858cd35cec6d


To use the model 

```
from AnyLengthBertModelAndUtils import SentencePairClassifier, MaskedGlobalAvgPool1d, MaskedGlobalMaxPool1d
```
```
from AnyLengthBertModelAndUtils import CustomDataset
```

For preproccessing text pairs, pass a dataframe of textpairs to CustomDataset, with columns Claim, Evidence. if labels are supplied, they should be under column label. And if they are not supplied the argument `with_labels` must be set to false, in instantiation

Example, loading data
```
test_set = CustomDataset(df_test,bert_model='bert-base-uncased',maxlen=128)
test_loader = DataLoader(test_set, batch_size=32, num_workers=2,collate_fn=my_collate)
```

Example loading model from .pt
```
model = torch.load('AnyLengthBert.pt')
```
note that SentencePairClassifier, MaskedGlobalAvgPool1d, MaskedGlobalMaxPool1d all must be imported into and accesable in main for the loading to work.

Example instantiating
```
net = SentencePairClassifier(bert_model='bert-base-uncased',freeze_bert=False)
```
only SentencePairClassifier must be imported.  MaskedGlobalAvgPool1d, MaskedGlobalMaxPool1d do not need importing if instantiating a new model.


How it works:

text is tokenised, and broken into overlapping chunks. These chunks are then fed one by one to a bert layer, the pooler outputs (tanh + dense layer on cls token embedding ) are then aggregated by  average pooling and max  pooling across them, then the average and max pools are summed, and then normalised. This is to get the benefit of maxpool which can be good at highlighting prominent features, with the more holistic and smoother average pooling.

AnyLengthBert.pt can be downloaded from https://www.dropbox.com/scl/fi/u1m76jo7ai2vd5kbk1kym/AnyLengthBert.pt?rlkey=qhdw3f5yqw8qzmfslbz795qqx&st=wixhnsh2&dl=1
But it takes only 3 minutes to train on the training dataset, with an L4 GPU



