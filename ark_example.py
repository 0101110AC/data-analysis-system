import os
from volcenginesdkarkruntime import Ark
# 从环境变量中读取您的方舟API Key
client = Ark(api_key=os.environ.get("ARK_API_KEY"))
completion = client.chat.completions.create(
    # 替换 <Model>为 Model ID
    model="doubao-1-5-lite-32k-250115",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
print(completion.choices[0].message)