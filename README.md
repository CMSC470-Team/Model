# Creative-Model-Name

Creating a question-answering model for our CMSC470 project which improves guess accuracy.

## Our Plan

1. Generate POS tagging for each questions

2. Generate NER for each question

3. For named entitiy in each question:

    * Use knowledge graphs to find the top k commonly linked entities

    * Transformer-XH

        * https://openreview.net/pdf?id=r1eIiCNYwS

        * https://github.com/microsoft/Transformer-XH
    
    * DELFT

        * https://github.com/henryzhao5852/DELFT

        * https://github.com/henryzhao5852/DELFT/tree/master/wiki_graph
   
   * QA-GNN
   
        * https://github.com/michiyasunaga/qagnn
        
        * https://arxiv.org/abs/2104.06378


4. Use 1-3 as features

6. Think of more features?

5. Choose some base model to place the features

    * Deep Averaging Networks

    * BERT

    * ...

## Current progress

   * Guesser
      
      Instead of using TF-IDF as the guesser, we tried to use different embedding techniques to find the answer for a specific query. We especially want to try sentence embedding, which represents entire sentences and their semantic information as vectors. Since our model gets questions chunk by chunk, we think sentence embedding can save computational power, since we don't need to encode every word and it can also capture the necessary context, intention, or other information in the text.
      
      In the bert.py file, we are currently using the sentenceBERT model ('all-mpnet-base-v2') to encode the sentences. Using the same logistic regression and same features, we found an increase compare to TF-IDF in the testing data.
      
      Accuracy for TF-IDF:
      ![alt text](https://github.com/CMSC470-Team/Model/blob/main/image/TF-IDF.jpg?raw=true)
      
      Accuracy for sentenceBERT:
      ![alt text](https://github.com/CMSC470-Team/Model/blob/main/image/BERT.png?raw=true)


## Dependencies

Check requirements.txt for the dependencies.
