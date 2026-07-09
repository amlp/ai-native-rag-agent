import requests

url = "https://coder.cnki.net/KCoder-Claude"
# 填入你的完整密钥
api_key = "MTY0NDEmJmQ0MWQ4Y2Q5OGYwMGIyMDRlOTgwMDk5OGVjZjg0Mjdl"
headers = {
    "Token": api_key,
    "Content-Type": "application/json"
}
body = {
    "model": "claude",
    "messages": [{"role": "user", "content": "测试接口连通性"}]
}
resp = requests.post(url, headers=headers, json=body)
print("状态码：", resp.status_code)
print("返回内容：", resp.text)