## babyllama项目

此项目基于[llama](https://github.com/Meta-Llama/llama)代码构建，使用的[llama-chinese](https://github.com/LlamaFamily/Llama-Chinese)中训练的tokenizer做分词。  
此外还参考了[babyllama](https://github.com/DLLXW/baby-llama2-chinese)对模型大小的研究，发现即使小模型（92M）在垂直领域微调也能有一定的效果。


在此基础上，我们构建了一个更小的模型（dim=256,layer=12,head=8），其参数量只有43M，在单卡RTX3090上即可训练，GTX1660ti上即可推理。  
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

在三个数据集上重新微调，然后在推理中加入top p增加随机性，发现模型倾向于重复说话，怀疑pCLUE所占比重太大且为单对话，回答内容没有参考，强行让模型模型记一些知识，虽然能处理一些多任务，但回答基本是错误的。  
<img src="assert\chat_2.png" alt="图片说明" width=100%>
<img src="assert\chat_3.png" alt="图片说明" width=100%>

NaturalConv数据集增加了模型基本聊天能力，但是在pCLUE数据集上训练后表现不行，并且数据集中回答通常为固定短语，对模型影响较大。  
删除pCLUE数据集重新训练试试。  

  
下一步，收集一些能增强模型逻辑能力的数据集，如数学，情感，脑筋急转弯等试试，然后去看一下语言模型的评价指标有哪些，看看能达到什么水平。  
  
评价指标
- 训练困惑度（Training Perplexity，通常缩写为 PPL），用于反应模型预测下一个词的准确率，PPL在训练中下降表示模型还有能力继续学习。公式为：
$$\text { Train PPL }=\exp \left(-\frac{1}{N} \sum_{i=1}^{N} \log P\left(w_{i}\right)\right)$$
- 代码能力，[HumanEval](https://github.com/openai/human-eval)。
- 常识推理，[CommonSenseQA ](https://github.com/jonathanherzig/commonsenseqa)
- 知识记忆，[NaturalQuestion](https://ai.google.com/research/NaturalQuestions/)，[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)
- 阅读理解 [QuAC](http://quac.ai/)，[BoolQ](https://github.com/google-research-datasets/boolean-questions)
- 数学能力 [GSM8k](https://github.com/openai/grade-school-math)，[MATH](https://github.com/hendrycks/math/)
- 多任务 [MMLU](https://huggingface.co/datasets/cais/mmlu)，[Big Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)，[AGIEval](https://github.com/ruixiangcui/AGIEval)  
  
  
咱们模型小，只能挑选部分感兴趣来做测试，并且都为英文得翻译一下，llama中使用zero-shot、few-shot来评价，说不定咱还得分点数据做微调。  
小模型就主要侧重逻辑推理能力和基本的对话，至于模型的知识，只能依靠检索的方式。  
暂定评价模型的数学，阅读理解和尝试推理。


[BelleGroup0.5M](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)数据集，由ChatGpt生成的数据集。  
[deepctrl-sft-data](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files)10M条数据的中文数据，50类任务。

添加了评价指标精度和困惑度  
现在尝试使用维基百科的数据预训练，然后在RedGpt、NaturalConv和BelleGroup0.5M上微调，也可以尝试在deepctrl-sft-data上微调。

在wiki语料上预训练的情况如下图所示  
<img src="assert\chat_2.png" alt="图片说明" width=100%>