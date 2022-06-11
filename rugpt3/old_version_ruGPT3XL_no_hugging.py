import re
from bs4 import BeautifulSoup
import pandas as pd

!pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# %%writefile setup.sh
# 
# export CUDA_HOME=/usr/local/cuda-10.1
# git clone https://github.com/NVIDIA/apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

!sh setup.sh

!apt-get install llvm-9-dev

!pip install cpufeature

!pip install triton==0.2.3

!python -m pip install --upgrade pip

! DS_BUILD_CPU_ADAM=1 && DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.3.7

!pip install torch==1.8.1+cu101 torchvision==0.9.0+cu101 torchtext==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

import deepspeed.ops.sparse_attention.sparse_attn_op

!git clone  https://github.com/sberbank-ai/ru-gpts

!pip install transformers==3.5.1

!pip install natsort

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("ru-gpts/")

import os
os.environ["USE_DEEPSPEED"] = "1"

from src.xl_wrapper import RuGPT3XL


gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)




def change_all_user_fields_qa(text):
  soup = BeautifulSoup(text)
  x = soup.body.findAll('span')
  final_text = ""
  miss = False
  last = ""
  for i in range(5):
    if miss==False:
      subsoup = BeautifulSoup(str(x[i]))
      replica = subsoup.body.find('span').text

      subsoup_next = BeautifulSoup(str(x[i+1]))
      replica_next = subsoup_next.body.find('span').text

      if ((replica[13:14]=="1" and replica_next[13:14]=="1") or (replica[13:14]=="2" and replica_next[13:14]=="2")):
        if (replica[13:14]=="1"):
          final_text+= "Q:" + replica[15:]  + ". " + replica_next[15:] + "\n"
          last = replica[13:14]
        elif (replica[13:14]=="2"):
          final_text+= "A:" + replica[15:]  + ". " + replica_next[15:] + "\n"
          last = replica[13:14]
        miss = True
      else:
        if (replica[13:14]=="1"):
          final_text+= "Q:" + replica[15:]  + "\n"
          last = replica[13:14]
        elif (replica[13:14]=="2"):
          final_text+= "A:" + replica[15:]  + "\n"
          last = replica[13:14]
    else:
      miss=False
  if last == "1":
    return final_text + "A:"
  elif last == "2":
    return final_text + "Q:"

dataset = pd.read_table('dialogues.tsv')

results_model_sber = []
dataset = dataset[0:2]

def filter_resuls(nr):
    return [x[:x.find("<|endoftext|>")] for x in nr]

for input, output in dataset.iterrows():
  input_temp = change_all_user_fields_qa(output["dialogue"])
  result =filter_resuls(gpt.generate(
    input_temp,
    top_k=5,
    top_p=0.95,
    temperature=1.2,
    num_return_sequences=5,
    max_length=max(40, len(output["dialogue"])*2),
    no_repeat_ngram_size=4,
    repetition_penalty=2.,
    do_sample=True,
    ))
  results_model_sber.append([item.replace(input_temp, '') for item in result])