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

## Dependencies

Check requirements.txt for the dependencies.
