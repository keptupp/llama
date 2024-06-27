## babyllama项目

此项目基于[llama](https://github.com/Meta-Llama/llama)代码构建，使用的[llama-chinese](https://github.com/LlamaFamily/Llama-Chinese)中训练的tokenizer做分词。  
此外还参考了[babyllama](https://github.com/DLLXW/baby-llama2-chinese)对模型大小的研究，发现即使小模型（92M）在垂直领域微调也能有一定的效果。


在此基础上，我们构建了一个更小的模型（dim=256,layer=12,head=8），其参数量只有43M。  
这个模型主要用于构建数据集和训练代码测试用，我们首先收集了[MNBVC](https://github.com/esbatmop/MNBVC)中的小部分数据做预训练（具体来说只下载了一个知乎新闻josnl，300MB）。  
基于这个预训练模型，我们收集了[RedGpt](https://github.com/DA-southampton/RedGPT)数据集作为对话的微调，其对话结构更改为：  
{bos 资料：xxxx。n\n\人类：xxxxx。n\n\助手：xxxx。 eos n\n\人类：xxxxx。n\n\助手：xxxx。 eos}  
按照此结构丢进模型进行训练。
最终训练效果不错，在模型推理过程中，我们使用wikipedia库收集维基百科数据作为资料，然后在此基础上对模型进行提问。  
如图：  
<img src="assert\chat_1.png" alt="图片说明" width=100%>
上述以及是表现比较好的时候，但是依然出现错误，说明模型对于文档的理解还是不够，并且测试的时候有可能出现反复重复短语。  
  
进一步的，收集了[NaturalConv](https://ai.tencent.com/ailab/nlp/dialogue/#datasets)数据集，训练模型日常聊天接话的能力（多对话）。  
[pCLUE](https://github.com/CLUEbenchmark/pCLUE)数据集，一个多任务数据集（单对话），涉及摘要，选择等具体项目。  

在三个数据集上重新微调，然后在推理中加入top p增加随机性。  
<img src="assert\chat_2.png" alt="图片说明" width=100%>
<img src="assert\chat_3.png" alt="图片说明" width=100%>

NaturalConv数据集增加了模型基本聊天能力，但是在pCLUE数据集上训练后表现不行，并且数据集中回答通常为固定短语，对模型影响较大。  

下一步，收集一些能增强模型逻辑能力的数据集，如数学，情感，脑筋急转弯等试试，然后去看一下语言模型的评价指标有哪些，看看能达到什么水平。  
  
[BelleGroup0.5M](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)数据集，由ChatGpt生成的数据集。
