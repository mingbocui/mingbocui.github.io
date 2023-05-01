---
title: "Two ways of finetuning LLM"
date: 2023-04-20T19:07:38+02:00
draft: false
mathjax: true
ShowCodeCopyButtons: true
---

Fine-tuning Large Language Models (LLMs) has become a significant focus as more individuals aim to utilize these models for specialized tasks. There are three primary approaches to consider when adapting an LLM for a specific subtask: LoRA, Adapters, and Prompt Engineering. However, prompt engineering is less scalable. Here only the first two methods will be discussed.

## LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

**LoRA**[^1] freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. Unlike adapters, no additional inference latency.


### Notions

${\Phi}_0$: pre-trained model weights  
$\Delta{\Phi}$: the weight changes during finetuning  
$\Delta W$: accuulated gradient update during adaptation  
$r$: rank of LoRA module  

**Intrinsic dimension**: Intrinsic dimension refers to the number of degrees of freedom or independent factors needed to describe a dataset or a model's behavior. In the context of pre-trained language models, a low intrinsic dimension means that the models can be represented with fewer parameters than their actual size, while still retaining most of their performance.

### How it works

Fine-tuning LLMs can be approached in a naive manner by performing backpropagation on all parameters or limiting it to specific layers, such as the last two layers. During full fine-tuning, the model's initialization uses pre-trained weights ${\Phi}_0$ and updates them to ${\Phi}_0 + \Delta{\Phi}$. A major drawback of full fine-tuning is that each downstream task requires learning a different set of parameters $\Delta {\Phi}$ with a dimension $|\Delta {\Phi}|$ equal to $|{\Phi}_0|$.

Pre-trained LMs like GPT and BERT possess a low "intrinsic dimension," allowing for efficient learning and compression within a smaller subspace without significantly sacrificing expressiveness.

![img](/paper_bites/lora_1.png#center)

Given a pre-trained weight matrix $W_0 \in R^{d×k}$, updates are constrained with a low-rank decomposition $W_0 +\Delta W = W_0 + BA$, where $B \in R^{d×r}$, $A \in R^{r×k}$, and the rank $r \ll min(d, k)$. During training, $W_0$ remains fixed without receiving gradient updates, while $A$ and $B$ have trainable parameters. Note that both $W_0$ and $\Delta W = BA$ are multiplied with the same input, with their output vectors summed coordinate-wise. For $h = W_0 x$, the modified forward pass is: $h = W_0 x + \Delta W x = W_0 x + BA x$.

To initialize matrices A and B, use random Gaussian initialization for A and zero for B. 

```py
nn.init.zeros_(self.lora_A)
nn.init.normal_(self.lora_B)
```

Consequently, $\Delta W = BA$ is zero at training onset. Next, scale $\Delta Wx$ by $\frac{\alpha}{r}$, with $\alpha$ as a constant in $r$. When optimizing with Adam, tuning $\alpha$ is similar to tuning the learning rate, provided the initialization is scaled accordingly. Thus, simply set $\alpha$ to the first $r$ tried, without further tuning.

```py
# here the self.weight refers to the weight of nn.Embedding
self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
self.scaling = self.lora_alpha / self.r

# the weights of BA 
self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
```

LoRA enhances training efficiency, as it eliminates the need to calculate gradients or maintain optimizer states for most parameters.


LoRA's simple design offers practical benefits and can be applied to any dense layers in deep learning models. Although the focus is on specific weights in Transformer language models for experimental purposes, the principles are widely applicable.

LoRA is a generalization of traditional fine-tuning, with the advantage of not requiring full-rank gradient updates to weight matrices during adaptation. By applying LoRA to all weight matrices and training all biases, the expressiveness of full fine-tuning is nearly recovered by setting the LoRA rank $r$ to the rank of the pre-trained weight matrices. As the number of trainable parameters increases, training LoRA converges towards training the original model, while adapter-based methods converge to an MLP and prefix-based methods to a model with limited input sequence length.

Inference latency is not increased with LoRA, as the explicit computation and storage of $W = W_0 + BA$ allows for standard inference. Transitioning to another downstream task involves quick operations with minimal memory overhead by subtracting $BA$ and adding a different $B'A'$.

In the Transformer architecture, there are four weight matrices in the self-attention module ($W_q$, $W_k$, $W_v$, $W_o$) and two in the MLP module. Attention weights are adapted for downstream tasks, while MLP modules are frozen for simplicity and parameter efficiency.

For large Transformers trained with Adam, VRAM usage is significantly reduced, especially when $r \ll d_{model}$. This results in up to a 2/3 reduction in VRAM consumption during GPT-3 175B training, from **1.2TB** to **350GB**. With $r = 4$, only query and value projection matrices are adapted, reducing the checkpoint size by around 10,000× (from 350GB to 35MB). This allows for training with fewer GPUs, avoids I/O bottlenecks, and results in a 25% speedup during GPT-3 175B training, as gradient calculations are not needed for most parameters.


## Adapter: Parameter-Efficient Transfer Learning for NLP

Transfer with adapter[^2] modules is an alternative approach that results in a compact and extensible model. Adapters add only a few trainable parameters per task and allow the addition of new tasks without revisiting previous ones. The original network parameters remain fixed, promoting significant parameter sharing. Adapter modules are characterized by a small number of parameters and a near-identity initialization.

### How it works

![](/paper_bites/adapter_1.png#center)

As shown in the above figure, a bottleneck architecture is proposed to limit the number of parameters. Adapters project the original $d$-dimensional features into a smaller dimension, $m$, apply a nonlinearity, and then project back to $d$ dimensions. The total number of parameters added per layer, including biases, is $2md + d + m$.

The corresponding code implementation is shown as below:


```py
with tf.variable_scope("adapters"):
    in_size = input_tensor.get_shape().as_list()[1]
    w1 = tf.get_variable(
        "weights1", [in_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b1 = tf.get_variable(
        "biases1", [1, hidden_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(input_tensor, w1, [[1], [0]]) + b1

    # add non-linearity
    net = gelu(net)

    w2 = tf.get_variable(
        "weights2", [hidden_size, in_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b2 = tf.get_variable(
        "biases2", [1, in_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(net, w2, [[1], [0]]) + b2

return net + input_tensor
```

### Application: LLaMA-Adapter

Since the open-sourcing of Meta's LLaMA[^3], various models have been engineered from it. LLaMA-Adapter[^4] is a notable example, providing a lightweight adaptation method to efficiently fine-tune LLaMA into an instruction-following model. Utilizing 52K self-instruct demonstrations, LLaMA-Adapter introduces merely 1.2M learnable parameters on top of the frozen LLaMA 7B model. Fine-tuning requires less than an hour on 8 A100 GPUs.

![](/paper_bites/llama_adapter_1.png#center)



## LoRA VS Adapter

Adapter is another way to finetune the LLM efficiently with just few parameters. However, large neural networks rely on hardware parallelism to keep the latency low, and adapter layers have to be processed **sequentially**. This makes a difference in the online inference setting where the batch size is typically as small as one. we see a noticeable increase in latency when using adapters, even with a very small bottleneck dimension.

![](/paper_bites/lora_2.png)



[^1]: Edward J. Hu, et al. "LoRA: Low-Rank Adaptation of Large Language Models." (2021). 
[^2]: Neil Houlsby, et al. "Parameter-Efficient Transfer Learning for NLP". CoRR abs/1902.00751. (2019).
[^3]: Hugo Touvron, et al. "LLaMA: Open and Efficient Foundation Language Models." (2023). 
[^4]: Renrui Zhang, et al. "LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention." (2023). 
