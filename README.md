# FewShotLLM_ClusterHypothesis
The final project of the course advanced IR in the topic of Few Shot LLM for Cluster Hypothesis Test


In addition to the requeirement.txt file, you might have to also run the command:
"pip install transformers, ir_datasets, tiktoken, openai, pyserini, faiss-gpu, torch==2.5.1".


We used the ranking library: "https://github.com/ielab/llm-rankers" to train our ranking model. the files from this library are in the folder "ranker library". The files at most are copied as they are from the original code, except we added some code to differ between the operation of the ranking and the clustering, and some altrecation in the prompts.
For further understaning the code in the files is documented.

By running the file main(after adjusting the correct hyperparameter) the corresponding model will be trained and evaluated on the data. You can observe the result in the folder created for it in the evalutation_measures file.
*** To run all files of the github must be in the same folder
