
# PLP-PM-2022-11-07-GRP-01-Virtual_Personal_Assistant

This is a virtual personal assistant to help users summarize and classify news that they subscribe to and respond to users' questions that are related to the news.




## Deployment

1. Create a python 3.7 environment as shown in the following. You can replace the <env> with any name you want to use.
```bash
conda create -n <env> python=3.7
conda activate <env>
```

2. In the code folder, run the below commands to install the packages.
```bash
pip install -f --user numpy pandas nltk bs4 python-telegram-bot spacy==2.3.5 tensorflow keras seaborn wordcloud scipy transformers SentencePiece sklearn chatterbot==1.0.4 chatterbot_corpus 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m spacy download en
conda install -y numpy jupyter notebook
```

3. Go to the question_answering folder, 
download the dataset zip file from https://drive.google.com/file/d/117wHqq5Cb4wDjqrMr0uXQ_FbBQSFThxZ/view?usp=sharing

download the model zip file from https://drive.google.com/file/d/1PvJPqAWSD0zygI-cBWkHn2YsZOVou0E1/view?usp=sharing

And unzip these two files into the question_answering folder.

4. Download the glove file from https://nlp.stanford.edu/data/glove.840B.300d.zip
And unzip the file “glove.840B.300d.txt” into dataset folder

5. Go to the text_summarization folder, download the dataset zip file from 
https://drive.google.com/file/d/1D6gKuxjNpjVU_KB16kaSI7-p75uwQQ2h/view?usp=sharing

And unzip the file into text_summarization folder

6. Run the python file server.py under telegram_bot folder to start the backend service