# KDD CUP 2020: Multimodalities Recall
### Team: WinnieTheBest

### Introduction
***
+ The competition prepares the real-scenario multimodal data from a mobile e-commerce platform. The dataset consists of searching queries and product image features, which is organized into a query-based multimodal retrieval task. We are required to implement a model to rank a collection of candidate products based on their image features. Most of these queries are noun phrases searching for products with specific characteristics. The images of the candidate products are provided by the seller's displaying photos, which are transformed into 2048-dimension features by a black box function. The most relevant candidate products to the query are regarded as the ground truth of the query, which are expected to be top-ranked by the participating models.

### Preprocess
***
+ Text
    + In `testA.tsv` we find many querys with weird grammar such as be verb to be the first word, which will harm the performance of the contextualized embedding. Hence we filter out such words to make a query more like a sentence.
+ Image Feature
    + Box feature
        + We concat 6-dimension box feature to orignal feature.
            + The first 4 dimensions are the given box position normalized to `[0,1]`
            + The other 2 dimensions are normalized area and aspect ratio.
    + Label feature
        + We also concat 32-dimension label feature, which is trainable with an embedding layer.

### Model Architecture & Parameters
***
+ We implement two different models:
    + MCAN[1]
        + The model consists of multiple MCA layers cascaded in depth to gradually refine the attended image and question features.
            + The MCA layer is a modular composition of the two basic attention units: the self-attention (SA) unit and the guided-attention (GA) unit, utilizing the scaled dot-product attention.
            + First we replace the word embeddings part in the cited paper with RoBERTa's word embeddings layer.
            + Second instead of adding the two flatten features of image and text, we concat the two flatten features. Moreover, to capture more information, we additionally concat the multiplication and $norm_1$ distance of the two flatten features.
        + Parameters
            + Pretrained model: roberta-large
            + LABEL_EMBED_SIZE = 32
            + BOX_SIZE = 6
            + IMG_FEAT_SIZE = 2048+BOX_SIZE+LABEL_EMBED_SIZE
            + WORD_EMBED_SIZE = 1024
            + LAYER = 12
            + HIDDEN_SIZE = 1024
            + MULTI_HEAD = 16
            + DROPOUT_R = 0.1
            + FLAT_MLP_SIZE = 512
            + FLAT_GLIMPSES = 1
            + FLAT_OUT_SIZE = 2048
            + FF_SIZE = HIDDEN_SIZE*4
    + VisualBERT[2]
        + Image regions and language are combined with a Transformer to allow the self-attention to discover implicit alignments between language and vision.
            + Different from the cited paper, to distinguish between the image features and language features we use `token_type_ids embeddings` with `[SEP]` token to separate the two features.
        + Parameters
            + Pretrained model: bert-large-uncased
            + BOX_SIZE = 6
            + IMG_FEAT_SIZE = 2048+BOX_SIZE

### Training Procedure
***
+ Negative Sampling
    + positive : negative = 1 : 10(k)
    + From `valid.tsv` we can see that the candidates of a query usually have similar words. We thus give higher sampling probability to the querys sharing same words with the positive query. However, it is not easy to tune the probability distribution. Also, another annoying thing is that the sampling ratio of the similar querys can be neither too high nor too low. Therefore, we come up with an easily-tuned method:
        + First we find the `topk` most similar querys sharing most words for each query
        + Second for each query, we sample querys from its `topk` most similar querys with the same amount of the number of the query.
        + Third for each sampled query, we uniformly draw one feature to be negative.
        + Finally repeat the second and the third step k times.
    + By the above method, all we need to do is tune `topk`. And we set `topk = max({numbers of features of querys})*3`
+ Training with the following parameters and schedule:
    + Learning rate = 1e-5
    + Batch size = 64
    + Loss: Focal Loss
    + Optimizer: AdamW
    + Scheduler: Sine Wave with Linear Warmup
        + At first we use linear schedule, but we find that the performance will be struck for several epochs during the last period of training. As a result, we modify the scheduler to be sine wave and it turns out that the valid score increases stably.
        + num_warmup_steps = num_training_steps*0.1
        + num_cycles = 6
        + amplitude = 0.3

### Postprocess
***
+ Re-train Classifier on `valid.tsv`
    + Since the above training process doesn't use the information of `valid.tsv`, which is the only ground truth we have. Therefore, we extract the embedding from the trained model to be the new classifier's input, then use `0/1` as training target to generate our final prediction.
    + Here we use `LightGBM` as this new classifier with all default hyperparameters.
+ Observation from `valid.tsv`
    + From `valid.tsv` we can see that the product with less appearances among the `valid.tsv` has higher probability to be the answer. We thus only keep the products which only occur once.
+ Ensemble
    + Totally we have 55 MCANs and 14 VisualBERTs, we simply add up the predicted logits to ensemble all models.

### Reproducibility
***
+ Requirement
    + `Python==3.8`
    + `torch==1.4.0`
    + `transformers==2.9.0`
    + `gensim==3.8.3`
    + `lightgbm==2.3.1`
+ Training
    + Run `share_master.ipynb` before run `MCAN-RoBERTa_pair-cat_box_tfidf-neg_focal_all_shared.ipynb` because our MCAN using shared memory.
    + `Visual-BERT_pair_box_tfidf-neg_focal_all.ipynb` can be run directly.
+ Prediction
    + Set `gpu_id` and `n_workers` for LightGBM as below:
        + `python3 MCAN-RoBERTa_pair-cat_box_tfidf-neg_focal_all_predict-all_cls.py {gpu_id} {n_workers}`
        + `python3 Visual-BERT_pair_box_tfidf-neg_focal_all_predict-all_cls.py {gpu_id} {n_workers}`
    + Run `./main.sh`
    + Note that with `n_workers = 24` it takes around 5 hours to predict.
### Follow up questions from issues:
> 1. Can you give a simple example of the negative sample sampling method?

Let the query pool be `['a cute dog', 'a cute bear', 'korean style of cat', 'japanese little dog', 'whatever it is']` and `topk = 4`. Then for query `'a cute dog'` we have an array of similar word counts: `[3, 2, 0, 1, 0]`. After that, we sorted the querys by this array and filter out the target query. So the negative querys of the target query would be `['a cute bear', 'korean style of cat', 'japanese little dog']`. Next moving on to sampling image features. Let the target query have `n` image features, then we should sample `n*k` negatives, where `k` is the negative sampling rate. We simply sampled `n` querys from its negative querys `k` times, and for each querys we uniformly sampled one image feature. Here we can see that `topk` should at least be the maximum number of numbers of features of querys plus one.

> 2. In this competition, 69 models have been trained based on Mcan and Visual Bert methods. Do these 69 models have any differences, such as parameters, training samples, etc?

Since we had a large negative sampling pool with large `topk` parameter, the only differences were random seeds for all random parts, which should be diverse enough.

> 3. Before post-processing, a single model based on Mcan or visual Bert is used to evalute the NDCG@5 on valid.tsv. How much can be achieved?

For VisualBERT, it was around 0.69. As for MCAN, it was around 0.71.

> 4. In the post-processing stage, the valid set is used to train the model. How to evaluate the model?

K-fold cross-validation on `valid.tsv`, and simple blending is applied afterward.

> 5. After post-processing, how many Score can a single model achieve in testA?

There was no enough time for us to test on testA. But it was around 0.87-0.88 on `valid.tsv`.

### Reference
***
[1] Yu, Zhou, et al. "Deep modular co-attention networks for visual question answering." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.* 2019.
[2] Li, Liunian Harold, et al. "Visualbert: A simple and performant baseline for vision and language." *arXiv preprint arXiv:1908.03557* (2019).

### Acknowledgement
+ We appreciate the advice and supports from Prof. Shou-De Lin under grant number 109-2634-F-002-033 from Taiwan Ministry of Science and Technology (MOST) ("Advanced Technologies for Resource-constrained Deep Learning") , Microsoft Research Asia Collaborative Project Funding (2019), and computation resources from National Center for High-Performance Computing.

+ HOSTKEY is now offering GPU dedicated servers and VPS with NVIDIA GTX 1080/1080Ti and RTX 2080Ti that are capable of processing any task at hand. We offer GPU servers with pre-installed PyTorch, Keras, Theano and TensorFlow libraries for dataflow, machine learning and neural networks.

You can train your neural network using HOSTKEY GPU solution 10x cheaper than on AWS or Google Cloud Platform but with the same speed!
Research: https://medium.com/@hostkey/data-science-experts-from-catalyst-have-compared-the-time-and-monetary-investment-in-training-the-cab231bc67d0
