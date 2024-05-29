

## Optimization approaches and abbreviation

1. Even Batching (EB)
2. Loss Rescaling (LR)
3. Additional Pretraining (AP)
4. Cross Encoding (CE)
5. Task Head Dropout (TD)

## Ablation Study
If not specially stated, training are done using a batch size of 8, with dropout layer in task heads, and task head layers in `multitask_classifier_ram.py` 

### Baseline with task head structure

- `multitask-sequential-old-struct`: sequential training + structure in `multitask_classifier.py`
    ```python
    class MultitaskBERT(nn.Module):
    
        def __init__(self, config):
            # ...
            # sentiment layers
            self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.sentiment_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
            # paraphrase layers
            self.paraphrase_dropout_1 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.paraphrase_dropout_2 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.paraphrase_linear_1 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
            self.paraphrase_linear_2 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
            self.paraphrase_linear_interact = torch.nn.Linear(BERT_HIDDEN_SIZE, 1)
            # similarity layers
            self.similarity_dropout_1 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.similarity_dropout_2 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.similarity_linear_1 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
            self.similarity_linear_2 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
            self.similarity_linear_interact = torch.nn.Linear(BERT_HIDDEN_SIZE, 1)
    
        def predict_sentiment(self, input_ids, attention_mask):
            output = self.forward(input_ids, attention_mask)
            output = self.sentiment_dropout(output) # dropout added
            sentiment_logits = self.sentiment_linear(output)
            return sentiment_logits


        def predict_paraphrase(self,
                            input_ids_1, attention_mask_1,
                            input_ids_2, attention_mask_2):
            output_1 = self.forward(input_ids_1, attention_mask_1)
            output_2 = self.forward(input_ids_2, attention_mask_2)
            output_1 = self.paraphrase_dropout_1(output_1)
            output_2 = self.paraphrase_dropout_2(output_2)
    
            output_1 = self.paraphrase_linear_1(output_1)
            output_2 = self.paraphrase_linear_2(output_2)
    
            output = torch.mul(output_1, output_2)
            paraphrase_logits = self.paraphrase_linear_interact(output)
            return paraphrase_logits


        def predict_similarity(self,
                            input_ids_1, attention_mask_1,
                            input_ids_2, attention_mask_2):
            output_1 = self.forward(input_ids_1, attention_mask_1)
            output_2 = self.forward(input_ids_2, attention_mask_2)
            output_1 = self.similarity_dropout_1(output_1)
            output_2 = self.similarity_dropout_2(output_2)
    
            output_1 = self.similarity_linear_1(output_1)
            output_2 = self.similarity_linear_2(output_2)
            output = torch.mul(output_1, output_2)
            similarity_logits = self.similarity_linear_interact(output)
            return similarity_logits
    ```

- `multitask-sequential-ram-struct`: sequential training + structure in `multitask_classifier_ram.py`
  ```python
  class MultitaskBERT(nn.Module):
    def __init__(self, config):
        # sentiment layers
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # from ram98 ======
        self.paraphrase_linear_for_dot = nn.Linear(config.hidden_size, config.task_embedding_size)
        self.paraphrase_linear_for_dot_dropout = nn.Dropout(config.task_dropout_prob)
        self.paraphrase_final_linear = nn.Linear(config.hidden_size * 2 + config.task_embedding_size, 1)
        self.paraphrase_final_dropout = nn.Dropout(config.task_dropout_prob)
        self.similarity_linear = nn.Linear(config.hidden_size, config.task_embedding_size)
        self.similarity_dropout = nn.Dropout(config.task_dropout_prob)
  
    def predict_sentiment(self, input_ids, attention_mask):
        output = self.forward(input_ids, attention_mask)
        output = self.sentiment_dropout(output) # dropout added
        sentiment_logits = self.sentiment_linear(output)
        return sentiment_logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        emb_1 = self.forward(input_ids_1, attention_mask_1)
        emb_2 = self.forward(input_ids_2, attention_mask_2)
        emb_1 = self.paraphrase_linear_for_dot_dropout(emb_1) # dropout added
        emb_2 = self.paraphrase_linear_for_dot_dropout(emb_2) # dropout added
        output_1 = self.paraphrase_linear_for_dot(emb_1)
        output_2 = self.paraphrase_linear_for_dot(emb_2)
    
        output_cat = torch.cat((emb_1, emb_2, output_1 * output_2), dim=1)
        logits = self.paraphrase_final_linear(self.paraphrase_final_dropout(output_cat)).view(-1)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        emb_1 = self.forward(input_ids_1, attention_mask_1)
        emb_2 = self.forward(input_ids_2, attention_mask_2)
        emb_1 = self.similarity_dropout(emb_1)
        emb_2 = self.similarity_dropout(emb_2)
    
        output_1 = self.similarity_linear(emb_1)
        output_2 = self.similarity_linear(emb_2)
        logits = F.cosine_similarity(output_1, output_2) * 5.0
        return logits
    ```

### Even batching and loss rescaling

- `even_batch-multitask_results`: even batching
- `even_batch-rescale0.1-multitask_results`: even batching + loss rescale 0.1
- `even_batch-rescale0.5-multitask_results`: even batching + loss rescale 0.5

### pretraining

- `pretrain-even_batch-rescale0.1-multitask_results`: pretraining + even batching + loss rescale 0.1
- `pretrain-even_batch-rescale0.2-multitask_results`: pretraining + even batching + loss rescale 0.2
- `pretrain-even_batch-rescale0.3-multitask_results`: pretraining + even batching + loss rescale 0.3

### Cross encoding
- `pretrain-even_batch-rescale0.2-cross-sstdrop-multitask_results`: pretraining + even batching + loss rescale 0.2 + cross encode **with SST task drophout**
- `pretrain-even_batch-rescale0.2-cross-nodrop-multitask_results`: pretraining + even batching + loss rescale 0.2 + cross encode **without task drophout**

### Task head dropout with batch size **16**
- `pretrain-even_batch-rescale0.2-cross-alldrop-multitask_results`: pretraining + even batching + loss rescale 0.2 + cross encode **with all 3 task drophout**
- `pretrain-even_batch-rescale0.2-cross-nodrop-multitask_results`: pretraining + even batching + loss rescale 0.2 + cross encode **without task drophout**
- `pretrain-even_batch-rescale0.2-cross-sstdrop-multitask_results`: pretraining + even batching + loss rescale 0.2 + cross encode **with SST task drophout**
- `pretrain-even_batch-rescale0.3-cross-sstdrop-multitask_results`: pretraining + even batching + loss rescale 0.3 + cross encode **with SST task drophout**
- `pretrain-even_batch-rescale0.3-cross-alldrop-multitask_results`: pretraining + even batching + loss rescale 0.3 + cross encode **with all 3 task drophout**





Baseline Structure:

| Stucture | SST Auc | Para Auc | STS Auc | Avg        |
| :------: | :------ | -------- | ------- | ---------- |
|   old    | 0.4913  | 0.7477   | 0.4456  | **0.5615** |
|   ram    | 0.4441  | 0.7884   | 0.6829  | **0.6384** |

Performance on dev set (best auc), **batch size 8**:

|     Note      | Even Batching | Loss Rescaling | Additional Pretraining | Cross Encoding | Task Dropout | SST Auc | Para Auc | STS Auc | Avg        | Batch Size |
| :-----------: | :------------ | -------------- | ---------------------- | -------------- | ------------ | ------- | -------- | ------- | ---------- | ---------- |
|   old arch    | --            | --             | --                     | --             | √            | 0.4913  | 0.7477   | 0.4456  | **0.5615** | 8          |
| Baseline arch | --            | --             | --                     | --             | √            | 0.4441  | 0.7884   | 0.6829  | **0.6384** | 8          |
|               |               |                |                        |                |              |         |          |         |            |            |
|     Loss      | √             | --             | --                     | --             | √            | 0.4850  | 0.8097   | 0.6955  | **0.6634** | 8          |
|               | √             | 0.1            | --                     | --             | √            | 0.4922  | 0.7916   | 0.6727  | **0.6522** | 8          |
|               | √             | 0.5            | --                     | --             | √            | 0.4950  | 0.7972   | 0.6870  | **0.6597** | 8          |
|               |               |                |                        |                |              |         |          |         |            |            |
| Pre-training  | √             | --             | √                      | --             | √            | 0.5122  | 0.7984   | 0.7608  | **0.6905** | 8          |
|               | √             | 0.1            | √                      | --             | √            | 0.4822  | 0.8015   | 0.7915  | **0.6918** | 8          |
|               | √             | 0.2            | √                      | --             | √            | 0.4895  | 0.8031   | 0.7841  | **0.6922** | 8          |
|               | √             | 0.3            | √                      | --             | √            | 0.4941  | 0.8033   | 0.7798  | **0.6924** | 8          |
|               |               |                |                        |                |              |         |          |         |            |            |
| Cross Encode  | √             | --             | √                      | √              | --           | 0.5050  | 0.8905   | 0.8789  | **0.7581** | 8          |
|               | √             | 0.1            | √                      | √              | --           | 0.4850  | 0.8872   | 0.8802  | **0.7508** | 8          |
|               | √             | 0.2            | √                      | √              | --           | 0.5013  | 0.8798   | 0.8789  | **0.7533** | 8          |
|               | √             | 0.2            | √                      | √              | SST          | 0.5004  | 0.8872   | 0.8842  | **0.7573** | 8          |
|               |               |                |                        |                |              |         |          |         |            |            |
|   Batch 16    | √             | --             | √                      | √              | √            | 0.5077  | 0.8901   | 0.8841  | **0.7606** | 16         |
|               | √             | --             | √                      | √              | --           | 0.5122  | 0.8879   | 0.8797  | **0.7600** | 16         |
|               | √             | 0.2            | √                      | √              | √            | 0.4886  | 0.8866   | 0.8802  | **0.7545** | 16         |
|               | √             | 0.2            | √                      | √              | --           | 0.4950  | 0.8865   | 0.8911  | **0.7575** | 16         |
|               | √             | 0.3            | √                      | √              | √            | 0.5013  | 0.8884   | 0.8915  | **0.7604** | 16         |
|               | √             | 0.3            | √                      | √              | SST          | 0.5022  | 0.8867   | 0.8871  | **0.7587** | 16         |
|   Batch 32    |               |                |                        |                |              |         |          |         |            |            |
|    **GPU**    |               |                |                        |                |              |         |          |         |            |            |
|               | √             | 0.3            | √                      | √              | √            | 0.5013  | 0.8857   | 0.8878  | **0.7583** | 32         |
|   RTX 6000    | √             | --             | √                      | √              | √            | 0.5095  | 0.8908   | 0.8836  | **0.7613** | 32         |
|     V100      | √             | --             | √                      | √              | √            | 0.5195  | 0.8885   | 0.8821  | **0.7633** | 32         |
|     A100      | √             | --             | √                      | √              | √            | 0.5277  | 0.8885   | 0.8748  | **0.7637** | 32         |
|   RTX2080ti   | √             | --             | √                      | √              | √            | 0.5104  | 0.8912   | 0.8794  | **0.7603** | 32         |



