# FedLM：Federated Learning Meets Large Models
在此设计中，我们提供了一套完整的代码结构和简要说明，以支持联邦学习环境的模拟。

```
/fedLM

|-- main.py      # 主程序入口和参数解析

|-- fed_server.py   # 服务器端代码，负责模型聚合

|-- fed_client.py   # 客户端代码，负责本地训练

|-- model.py     # 定义CNN模型

|-- dataset.py    # 数据加载和预处理

|-- config.py     # 存放配置参数

|-- utils.py     # 辅助功能
```

### 记录

20240515： First Update, 简单实现了FedAvg，多个客户端联合训练CNN用于Mnist。
