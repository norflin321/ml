with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(len(text))
print(text[:1000])
