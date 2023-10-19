from transformers import pipeline

summarizer = pipeline("summarization", model="lxyuan/distilbart-finetuned-summarization")
with open("data/sample1.txt", "r") as f:
    sample = f.read().replace("\n", " ")
    s = summarizer(sample, min_length=50, max_length=250)
    print(s[0]['summary_text'])