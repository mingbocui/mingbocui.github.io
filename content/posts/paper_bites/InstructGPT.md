---
title: "Details of InstructGPT"
date: 2023-04-20T19:09:34+02:00
draft: false
mathjax: true
---

InstructGPT was developed to address the misalignment between the general objective of language modeling and human requirements. While large LMs focus on predicting the next token from internet web pages, humans desire models that "follow the user’s instructions helpfully and safely.” InstructGPT aligns the behavior of GPT-3 with the preferences of specific groups, such as labelers and researchers, rather than broader "human values."

This blog will first explore data collection and then delve into the details of each step of training the model.

# Data Collection

## Labelers selection

OpenAI employed a team of 40 contractors to label data, selecting them based on their performance in a **screening test**. The screening test aimed to identify labelers who were sensitive to the preferences of various demographic groups and adept at recognizing potentially harmful outputs. Meanwhile the labelers work closely with OpenAI, and guidelines are frequently updated for improvements.


## Data source

### Two sources of prompt:  
1. collected from OpenAI playground   
    - deduplicate prompts by checking for prompts that share a long common prefix
    - limit the number of prompts to 200 per user ID.
    - create the train, validation, and test splits based on user ID, the validation and test sets contain no data from users whose data is in the training set. 
    - to avoid the models learning potentially sensitive customer details, we filter all prompts in the training split for **personally identifiable information** (PII).

2. labelers self-written prompts

For each natural language prompt[^2], the task is most often specified directly through a natural language instruction (e.g. “Write a story about a wise frog”), but could also be indirectly through either few-shot examples (e.g. giving two examples of frog stories, and prompting the model to generate a new one) or implicit continuation (e.g. providing the start of a story about a frog). In each case:
- ask the labelers to do their best to **infer** the intent of the user who wrote the prompt
- ask them to **skip** inputs where the task is very unclear

### Source of demonstrations(responses) w.r.t prompts:  
- labelers write demonstrations for the prompts

From these prompts, three different datasets are produced in the fine-tuning procedure: 
- SFT dataset, with labeler demonstrations used to train the SFT models  
- RM dataset, with labeler rankings of model outputs used to train the RMs  
- PPO dataset, without any human labels, which are used as inputs for RLHF fine-tuning. 

An oveview of the three datasets used in each step is shown as below: 

![](/paper_bites/instructgpt_3.png#center)


# Model

The whole pipeline is composed of 3 steps:
![](/paper_bites/instructgpt_2.png#center)

## Step 1: supervised finetuning (SFT)  

### Dataset

To train the very first InstructGPT models, labelers were asked to **write prompts themselves**. This is because an initial source of instruction-like prompts to bootstrap the process is needed, and **these kinds of prompts weren’t often submitted to the regular GPT-3 models on the API**. OpenAI asked labelers to write three kinds of prompts[^2]:
- Plain: simply ask the labelers to come up with an arbitrary task, while ensuring the tasks had sufficient diversity.  
- Few-shot: ask the labelers to come up with an instruction, and multiple query/response pairs for that instruction.  
- User-based: there is a number of use-cases stated in **waitlist applications** to the OpenAI API. OpenAI asked labelers to come up with prompts corresponding to these use cases.  

**Prompt source**: OpenAI platform and labeler-written  
**Dataset size**: 13k  


### Model training

**Base model**: GPT3  
**Epochs**: 16 with learning rate decay  
**dropout**: 0.2  

The Step 1 will fine-tune GPT-3 on the labeler demonstrations using supervised learning. SFT models overfit on validation loss after 1 epoch, but training more epochs helps both the RM score and human preference ratings despite the overfitting. This model is not used directly but just for future use, so overfitting is not a problem.


## Step 2: Reward Model training  

Human-in-the-loop is expensive. The model cannot wait for human's real time preference score to proceed. So it would be good to have a model that can have the same preference taste as humans.

### Dataset

**Prompt source**: OpenAI platform and labeler-written 
**Dataset size**:  33k

To prepare the data for training Reward Model, first a prompt will be fed into the SFT model trained in Step 1. There would be several responses generated for one prompt. Then the labelers will rank these responses.  

Why we prepare the comparison data?  
- human judgments are noisy and miscalibrated, absolute score is not reliable
- pair of comparisons can be formed for more data

In Stiennon et al[^1], the RM is trained on a dataset of comparisons between **two** model outputs on the same input. They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler. Basically it means the model will try to maximize the score of the preferred output, which is binary classification.

To speed up the collection of comparisons data, labelers are asked to rank among $K$(4~9)  responses. This produces ${K \choose 2}$ pair comparisons for each prompt shown to a labeler. 

#### Why should not shuffle the comparisons into a single dataset 

Due to the high correlation of comparisons within each labeling task, shuffling the comparisons into a single dataset led to overfitting when the Reward Model performed a single pass over the dataset.

Let's consider one example, suppose we have K=4 responses for a given prompt. Labelers rank these responses, and there will be ${4 \choose 2} = 6$ possible pair comparisons:  
```
1. C1 vs C2
2. C1 vs C3
3. C1 vs C4
4. C2 vs C3
5. C2 vs C4
6. C3 vs C4
```

Notice that each completion (C1, C2, C3, and C4) appears in 3, namely $(K-1)$ comparisons. 

So if we treat each comparison as a separate data point and **shuffle** them together, the training dataset might look like this after shuffle:

```
1. C1 vs C2
2. C3 vs C4
3. C2 vs C4
4. C1 vs C3
5. C2 vs C3
6. C1 vs C4
```

When the model is trained on this dataset, it will process each comparison in sequence. As the model moves through the dataset, each completion will be used for gradient updates multiple times(or **$K-1$** times precisely). 

In this example, completion $C1$ is used for gradient updates in steps 1, 4, and 6. Since the comparisons involving the same completion are highly correlated, processing them separately during the same epoch can lead to overfitting. The model will learn to focus too much on the **specific characteristics of these completions**, and its performance on unseen data will suffer.

To address this issue, the authors propose training on all ${K \choose 2}$ comparisons from each prompt **as a single batch element**. This reduces the number of forward passes required and helps prevent overfitting, as the model learns from all comparisons at once rather than processing them individually.


### Model training

The original GPT model will have a softmax layer(unembedding layey) which will generate a probability for each token in the sentence. This step will remove this softmax layer and simply add a linear projection layer in order to **output a scalar as reward**. The Reward Model will take prompt and response as input, and a scalar will be the output.

**Base model**: GPT3-6B rather than 175B due to instablity

#### Notations

- $D$: dataset of human comparisons, one valid sample looks like $(x,\ y_w,\ y_l)$:
    - $x$: prompt  
    - $(y_w,\ y_l)$: two demonstrations for prompt $x$ that $y_w$ will have a higher score than $y_l$

- $r_{\theta}$: the Reward Model with $\theta$ as weights to be trained

#### Loss function

The objective of Reward Model is to train a set of weights ($\theta$) that can score high quality response $y_w$ with a higher score.


The loss function is:

> $loss({\theta}) = -\frac{1}{K \choose 2}E_{(x,\ y_w,\ y_l) \sim D}[log(\sigma(r_{\theta}(x,\ y_w)-r_{\theta}(x,\ y_l)))]$


As we discussed before, ${K \choose 2}$ comparisons are served as a single batch element, so $\frac{1}{K \choose 2}$ is a normalization coefficient to avoid model to be effected by the size of $K$. 

As the training finised, we could assume that the trained RM will replace the human and learned the human's preference (actually it is the **labeler's preference**). Then the Reward Model $r_{\theta}$ can replace human to score the outputs of models during the training.


## Step 3: RL via PPO(Proximal Policy Optimization)  

This step will optimize a policy against the Reward Model using PPO algorithm. The goal of this step is to finetune the SFT model on the environment using PPO, where the environment is a **bandit environment** which reprsents a random customer prompt and expects a respnse to the prompt.

### Dataset

**Prompt source**: only OpenAI platform  
**Dataset size**:  31k  

### Model training

Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best policy(the comparison data is scored by the trained RM), which is used to train a new RM and then a new policy. In practice, most of the comparison data comes from the supervised policies, with some coming from the PPO policies.

#### Notations
$\pi^{\text{SFT}}$: the supervised trained model from Step 1  

$\pi_{\phi}^{\text{RL}}$: the learned RL policy. It will be initialized from $\pi^{\text{SFT}}$ in the beginning  

$D_{pretrain}$: the distribution of pretraining data of GPT3  

$r_{\theta}$: the Reward Model trained from Step 2    

$(x,\ y)$: the prompt $x$ and a corresponding response $y$ from the model updated by RL policy  

${\pi_{\phi}^{\text{RL}}(y|x)}$: the probability distribution of response $y$ (per-token) given prompt $x$ under the policy, namely multiply the probability of each token  


#### Objective function

> $\text{objective}(\phi) = E_{(x,\ y) \sim D_{\pi_{\phi}^{\text{RL}}}}[r_{\theta}(x,\ y) - \beta \log(\frac{\pi_{\phi}^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)})] + \gamma E_{x \sim D_{\text{pretrain}}} [\log(\pi_{\phi}^{\text{RL}}(x))]$


Note that now the response $y$ will come from the updated model ${\pi_{\phi}^{\text{RL}}}$.
The train data of Reward Model $r_{\theta}$ in Step 2 could be differ from the $y$ generated from the ${\pi_{\phi}^{\text{RL}}}$, in that case the estimated score produced by 
$r_{\theta}$ could be unreliable.  
To prevent them differ too much, the author add this term **$\log(\frac{\pi_{\phi}^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)})$** (KL divergence term) to regulate the learned policy to not generate outputs that are too far away from those used to train the Reward Model. Otherwise we cannot trust the reward score generated by the Reward Model. 

Therefore the regularization term $\log(\frac{\pi_{\phi}^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)})$ is expected to be as little as possible. To include it into the objective function, a negative sign is put in the front of it so one can maximize $-\beta \log(\frac{\pi_{\phi}^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)})$



The last term $\gamma E_{x \sim D_{\text{pretrain}}} [\log(\pi_{\phi}^{\text{RL}}(x))]$, the term $\pi_{\phi}^{\text{RL}}(x)$ is the probability of generating propt $x$ with the updated model $\pi_{\phi}^{\text{RL}}$. If the $\gamma$ is greater than 0, the trained model is called **PPO-ptx** otherwise **PPO**.

From the figure below, one can observe the model trained with PPO policy is better than other models. And PPO-ptx is better than PPO model when the model size is big enough.

![](/paper_bites/instructgpt_1.png#center)



## Extra notes

OpenAI also disclosed some conflicted alignment during training and evaluation[^2]. For example, when a user requests a potentially harmful response. 
- During training they **prioritize helpfulness** to the user
- in final evaluations labelers are asked to **prioritize truthfulness and harmlessness** (since this is what they really care about)



[^1]: Nisan Stiennon, et al. "Learning to summarize from human feedback". CoRR abs/2009.01325. (2020).
[^2]: Long Ouyang, et al. "Training language models to follow instructions with human feedback." (2022). 