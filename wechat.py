import requests
headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjQ1MTA4NywidXVpZCI6ImQwYzhiYTA3LTllODgtNGE2Ny05OTNiLThlZGVmMWZmZGFhOCIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.y4FXv_RWZI7CbC1IHUnhwHnZJiHiLc_Fe6WKwCI2Iz2fgZjEiDRDpOBnr0XSKiUVfBJg4JHnalwIyiY3NSUiIg"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "多模态训练",
                         "name": "eg. 实验程序bug或结束",
                         "content": "eg. Epoch=100. Acc=90.2"
                     }, headers = headers)
print(resp.content.decode())