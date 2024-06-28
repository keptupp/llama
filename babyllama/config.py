import torch
device=torch.device("cuda")

import requests

# proxies = {
#     'http': '127.0.0.1:7890',
#     'https': '127.0.0.1:7890'
# }

# query='哆啦A梦是什么？'
# search_url = f"https://www.bing.com/search?q={query}"
# response = requests.get(search_url,proxies=proxies)

# print(response.text)