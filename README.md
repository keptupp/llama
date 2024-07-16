## babyllama项目


此项目基于[llama](https://github.com/Meta-Llama/llama)代码构建，使用的[llama-chinese](https://github.com/LlamaFamily/Llama-Chinese)中训练的tokenizer做分词。  
此外还参考了[babyllama](https://github.com/DLLXW/baby-llama2-chinese)对模型大小的研究，发现即使小模型（92M）在垂直领域微调也能有一定的效果。

### 知乎新闻预训练，对话微调
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

在三个数据集上重新微调epoch10的时候精度57.64%，然后在推理中加入top p增加随机性，发现模型倾向于重复说话，怀疑pCLUE所占比重太大且为单对话，回答内容没有参考，强行让模型模型记一些知识，虽然能处理一些多任务，但回答基本是错误的。  
<img src="assert\chat_2.png" alt="图片说明" width=100%>
<img src="assert\chat_3.png" alt="图片说明" width=100%>

NaturalConv数据集增加了模型基本聊天能力，但是在pCLUE数据集上训练后表现不行，并且数据集中回答通常为固定短语，对模型影响较大。  
删除pCLUE数据集重新训练试试。  


### 进一步收集的相关数据
  
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

### wiki数据预训练，对话微调

添加了评价指标精度和困惑度  
现在尝试使用维基百科的数据预训练，然后在RedGpt、NaturalConv和BelleGroup0.5M上微调，也可以尝试在deepctrl-sft-data上微调。

在wiki语料上预训练的情况如下图所示，只跑1个epoch的效果，大约花了7个小时，训练效果没有达到预期，模型还没完全学习到wiki中的知识，后续再加几个epoch   
<img src="assert\pre_train_1.png" alt="图片说明" width=100%>
<img src="assert\pre_train_2.png" alt="图片说明" width=100%>
<img src="assert\pre_train_3.png" alt="图片说明" width=100%>
后续一个epoch
<img src="assert\pre_train_4.png" alt="图片说明" width=100%>
<img src="assert\pre_train_5.png" alt="图片说明" width=100%>
<img src="assert\pre_train_6.png" alt="图片说明" width=100%>
  
下面尝试在redgpt数据集上微调 3 epoch精度57.8%，看起来模型说话的长度变多了，倾向于多介绍一些具体信息。
在BelleGroup0.5M上微调 3 epoch精度60.2%，可以处理一些简单的多任务，如讲一个笑话，取标题，编写文章等，但是没有连续对话的能力。  
<img src="assert\chat_4.png" alt="图片说明" width=80%>
<img src="assert\chat_5.png" alt="图片说明" width=80%>
<img src="assert\chat_6.png" alt="图片说明" width=30%>


### wiki权重后的deepctrl_sft微调
在deepctrl-sft-data上微调，与其说微调倒不如说更细致的预训练，这个数据集一共训练了50多小时，涵盖的任务广泛。  
数据太大，没有打乱数据，依次对整合的几个数据集进行了训练，使得模型学习到了一些更加广泛的能力，基本上是样样会，但不精。  
在loss图上可以清晰的看见在任务分布差异较大的时候loss的剧烈抖动。
<img src="assert\deepctrl_sft_7.png" alt="图片说明" width=100%>
<img src="assert\deepctrl_sft_8.png" alt="图片说明" width=100%>
<img src="assert\deepctrl_sft_9.png" alt="图片说明" width=100%>
在数据集上看了一下第一次loss抖动的时候，开始学习了翻译能力，其次还有代码编写能力等。
<img src="assert\chat_7.png" alt="图片说明" width=100%>
<img src="assert\chat_8.png" alt="图片说明" width=100%>
上述是表现比较好的情况，模型存在遗忘，对训练前期的任务有遗忘，并且这个数据集中也存在一些噪声和安全限制，如模型现在生成自己是Moss，还有一些简单的问题拒绝回答。  
就把这个数据集也当一个预训练挺好的，相比于wiki数据集的预训练这个数据集更大，预训练的指标表现均更好，说明单纯的让小模型去记忆一些毫无关联的知识数据是很困难的，相反直接训练一些复杂任务，能够提高模型的逻辑能力，具有更高的性价比。  
  
接下来进一步选取单一任务，成为一个能够基本对话和文本摘要能力的模型。  
看看怎么把lcsts和csl数据集进行微调并评分。

### 微调为摘要专家模型，同时兼具基本对话。
基本多对话数据集  
[BelleGroup0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)偏生成知识讨论，缺点是需要模型记住一些外部知识，好处是能够提高复杂知识的理解能力。  
NaturalConv好处是有基本的短对话，能满足基本聊天，缺点是聊天中经常出现没有解释的域外知识关键词进行讨论。  
LCSTS和CSL文本摘要数据集  
  
摘要数据集需要进行对话处理，具体实现方法是设置多个询问和回答模板，在模板中加入一些可替换的名词动词等，在数据集加载时动态随机的为每一条摘要数据集都加上询问和回答，从而实现对摘要数据集的对话转变。模板内容如下。    
<img src="assert\chat_12.png" alt="图片说明" width=50%>
  
使用BelleGroup0.8M和CSL数据集进行摘要专家模型的训练，3epoch训练过程如下  
<img src="assert\multi_chat_csl_10.png" alt="图片说明" width=100%>
<img src="assert\multi_chat_csl_11.png" alt="图片说明" width=100%>
<img src="assert\multi_chat_csl_12.png" alt="图片说明" width=100%>
对话表现如下，可看见在兼具基本聊天的情况下可以对文章内容进行总结，并且还兼具了进一步对文章具体信息提问的能力（现在效果不好，后续考虑加入RedGpt数据集，这个数据集对文章内容讨论的对话）。此时rouge-1f分数0.567  
<img src="assert\chat_10.png" alt="图片说明" width=100%>
<img src="assert\chat_9.png" alt="图片说明" width=100%>
与多任务大数据集预训练和多对话数据集进行对比下，加入摘要对话数据集后提升效果是显著的。如下为多任务大数据集上的表现（多对话更差）  
<img src="assert\chat_11.png" alt="图片说明" width=100%>

最终在CSL测试数据集上的rouge-1的f1分数为0.567，相比于我的文本摘要模型0.60低了3个多点。  
尝试只使用CLS数据集训练10个epoch，最终rouge-1f评分0.584


后续计划
将专家摘要对话中的摘要与原始摘要进行rouge评分  
使用lora的微调方法看看效果。   
找一个vqa数据集，尝试接入efficientvit网络（后续看看怎么特征融合）做多模态.