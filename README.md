## babyllama项目

此项目基于[llama](https://github.com/Meta-Llama/llama)代码构建，使用的[llama-chinese](https://github.com/LlamaFamily/Llama-Chinese)中训练的tokenizer做分词。  
此外还参考了[babyllama](https://github.com/DLLXW/baby-llama2-chinese)对模型大小的研究，发现即使小模型（92M）在垂直领域微调也能有一定的效果。


在此基础上，我们构建了一个更小的模型（dim=256,layer=12,head=8），其参数量只有43M。  
这个模型主要用于构建数据集和训练代码测试用，我们首先收集了[MNBVC](https://github.com/esbatmop/MNBVC)中的小部分数据做预训练（具体来说只下载了一个知乎新闻josnl，300MB）。  
基于这个预训练模型，我们收集了[RedGpt](https://github.com/DA-southampton/RedGPT)数据集作为对话的微调，其对话结构更改为：  
{bos 资料：xxxx。n\n\人类：xxxxx。n\n\助手：xxxx。 eos n\n\人类：xxxxx。n\n\助手：xxxx。 eos}  
按照此结构丢进模型进行训练。
最终训练效果不错，在模型推理过程中，我们使用wikipedia库收集维基百科数据作为资料，然后在此基础上对模型进行提问。