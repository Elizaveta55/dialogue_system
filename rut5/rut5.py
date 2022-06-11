import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-small-chitchat")
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small-chitchat")

sent1 = "Я пошел гулять."
sent2 = "Я чувствую обиду и злость за свой поступок."
sent3 = "Снег автомату рознь, а собака скользкая."
sent4 = "Скажи мне что-либо приятное."

dataset = [sent1,sent2,sent3,sent4]

for item in dataset:
  text = item
  inputs = tokenizer(text, return_tensors='pt')
  start_time = time.time()
  with torch.no_grad():
      hypotheses = model.generate(
          **inputs, 
          do_sample=True, top_p=0.5, num_return_sequences=1, 
          repetition_penalty=2.5,
          max_length=32,
      )
  print(item)
  print("calc time {}".format(time.time() - start_time))
  for h in hypotheses:
      print(tokenizer.decode(h, skip_special_tokens=True))