from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

import time

sent1 = "Q: Я пошел гулять. \n A: "
sent2 = "Q: Я чувствую обиду и злость за свой поступок. \n A:"
sent3 = "Q: Снег автомату рознь, а собака скользкая. \n A:"
sent4 = "Q: Скажи мне что-либо приятное. \n A:"

dataset = [sent1, sent2, sent3, sent4]

for item in dataset:
  text = item
  time_start = time.time()
  input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
  out = model.generate(
      input_ids.cuda(),
      top_k=1,
      top_p=0.95,
      temperature=1.2,
      num_return_sequences=1,
      max_length=32,
      no_repeat_ngram_size=3,
      repetition_penalty=2.5)
  generated_text = list(map(tokenizer.decode, out))[0]
  print("calc time {}".format(time.time() - time_start))
  print(generated_text)
  print("-----------------------------------")