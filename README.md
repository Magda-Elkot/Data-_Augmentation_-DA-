# Data Augmentation (DA)
## Brief Abstraction:
    Data Augmentation (DA) Technique is a process that enables us to artificially increase training data size by generating different versions of real datasets
    without collecting the data. Based on “Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks”. EDA consists of four
    simple but powerful operations: synonym replacement, random insertion, random swap, and random deletion. On five text classification tasks, we show
    that EDA improves performance for both convolutional and recurrent neural networks. EDA demonstrates particularly strong results for smaller datasets.
    Automatic data augmentation is commonly used in computer vision and speech and can help train more robust models, particularly when using smaller datasets.
    However, because it is challenging to come up with generalized rules for language transformation, universal data augmentation techniques in NLP have
    not been thoroughly explored. In this paper, a simple set of universal data augmentation techniques for NLP called EDA (easy data augmentation) is
    presented. They systematically evaluate EDA on five benchmark classification tasks, showing that EDA provides substantial improvements on all five tasks and
    is particularly helpful for smaller datasets.
    
## Benchmark Datasets:
  ### Based on the paper, we will Insha’Allah conduct experiments on five benchmark text classification tasks:
  
      (1) SST-2: Stanford Sentiment Treebank (Socher et al., 2013)
      
      (2) CR: customer reviews (Hu and Liu, 2004; Liu et al., 2015)
      
      (3) SUBJ: subjectivity/objectivity dataset (Pang and Lee, 2004)
      
      (4) TREC: question type dataset (Li and Roth, 2002)
      
      (5) PC: Pro-Con dataset (Ganapathibhotla and Liu, 2008).
      
## Models/Tools:
  ### We tested several augmentation operations loosely inspired by those used in computer vision and found that they helped train more robust models. 
  ### For a given sentence in the training set, we randomly choose and perform one of the following operations:
      1. Synonym Replacement (SR): Randomly choose n words from the
      sentence that are not stop words. Replace each of these words with one
      of its synonyms chosen at random.
      
      2. Random Insertion (RI): Find a random synonym of a random word in
      the sentence that is not a stop word. Insert that synonym into a random
      position in the sentence. Do this n times.
      
      3. Random Swap (RS): Randomly choose two words in the sentence and
      swap their positions. Do this n times.
      
      4. Random Deletion (RD): Randomly remove each word in the sentence
      with probability p.
      Since long sentences have more words than short ones, they can absorb more
      noise while maintaining their original class label. To overcome this, we vary the
      number of words changed (n), for SR, RI, and RS based on the sentence length
      (l) with the formula n=αl, where α is a parameter that indicates the percent of the
      words in a sentence are changed (we use p=α for RD). α=0.1 appeared to be a
      “sweet spot” across the overall experimentations implemented during the
      research feeding the information in that paper.
