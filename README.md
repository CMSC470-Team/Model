# Creative-Model-Name

## Authors: Abhay Patel, Jake Baldwin, Seyed Ghaemi, Zihan Ma 

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
   
   * NER as Feature
      
      We are using NLTK's named entity recognizer. Currently we tried simply adding NE as a feature to the tfidf_guesser with little improvement in accuracy. Our next plan is to find a Python graph library and use that in conjunction with [DELFT's wiki graph](https://github.com/henryzhao5852/DELFT/tree/master/wiki_graph) to calculate a score for the connections between the named entities in the graph. We will then use these scores to find the named entitiy most related to all the named entities in the question.  

   * Guesser
      
      Instead of using TF-IDF as the guesser, we tried to use different embedding techniques to find the answer for a specific query. We especially want to try sentence embedding, which represents entire sentences and their semantic information as vectors. Since our model gets questions chunk by chunk, we think sentence embedding can save computational power, since we don't need to encode every word and it can also capture the necessary context, intention, or other information in the text.
      
      In the bert.py file, we are currently using the sentenceBERT model ('all-mpnet-base-v2') to encode the sentences. Using the same logistic regression and same features, we found an increase compare to TF-IDF in the testing data.
      
      Accuracy for TF-IDF:
      ![alt text](https://github.com/CMSC470-Team/Model/blob/main/image/TF-IDF.jpg?raw=true)
      
      Accuracy for sentenceBERT:
      ![alt text](https://github.com/CMSC470-Team/Model/blob/main/image/BERT.png?raw=true)
      
      Fine tunned sentenceBERT model and required file
      https://drive.google.com/file/d/1ucp4FnZdnxj820JH_-ootlSm1Fv1A1BK/view?usp=sharing (unzip it)
      
      https://drive.google.com/file/d/1Xstdah7kYt7F14gqKEcgxl6xVE8ibRFJ/view?usp=sharing

## Dependencies

Check requirements.txt for the dependencies.
