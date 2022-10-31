# PLP-PM-2022-11-07-GRP-01-Virtual_Personal_Assistant

```console
conda create -n py37 python=3.7
conda activate py37
conda install numpy jupyter notebook
pip install --user numpy pandas nltk bs4 python-telegram-bot chatterbot==1.0.4 chatterbot_corpus spacy==2.3.5
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m spacy download en
```