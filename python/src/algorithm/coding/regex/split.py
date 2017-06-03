import re

S = input().strip()
S = re.split(r'[,.]+', S)

[print(i) for i in S if i]
