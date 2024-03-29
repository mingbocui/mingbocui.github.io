---
title: "HuggingGPT"
date: 2023-04-17T19:28:16+02:00
draft: false
---


## HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face


HuggingGPT[^1] is a framework that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. 

LLM has shown great abilities but it still confronts some **challenges**:
1. lack the ability to process complex information such as vision and speech
2. some complex real world tasks are usually composed of multiple sub-tasks, and require the scheduling and cooperation of multiple models, which are also beyond the capability of language models; 
3. for some challenging tasks, LLMs demonstrate excellent results in zero-shot or fewer shot results but they are still **weaker** than experts models(models fintuned on subtasks)


HuggingGPT can incorporate the expert model's descriptions into prompts and ask LLM to manage these expert models. Huggingface is a achine learning community which consists a lot of high quality models(400+ task-specific models) which specific for speech, vision and also language.

## Settings

**LLM**: gpt-3.5-turbo and text-davinci-003  
**decoding** temperature: 0  
**logit_bias**: 0.2 (format constraints)  

## Method

For each AI model in Hugging Face, we use its corresponding **model description** from the library and fuse it into the prompt to establish the connection with ChatGPT.   
There are **4** stages:

- **Task Planning**: Using ChatGPT to **analyze** the requests of users to understand their intention, and disassemble them into possible solvable tasks via prompts.  
    - Complex requests often involve multiple tasks, requiring the large language model to determine task dependencies and execution order. HuggingGPT uses **specification-based instruction** and **demonstration-based parsing** for effective task planning in the design of prompt:
        - specification-based instruction: A uniform template with four slots (`task type`, `task ID`, `task dependencies`, and `task arguments`) enables task parsing through slot filling.
        - demonstration-based parsing: In-context learning improves task parsing and planning by injecting demonstrations into prompts, helping the model better understand intentions and criteria.

- **Model Selection**: To solve the planned tasks, ChatGPT **selects** expert models that are hosted on Hugging Face based on model descriptions.
    - Limited context length restricts including all relevant model information in prompts. To resolve this, the author filter models by task type, retaining only matching models. These models are ranked by the number of downloads on the hugging face website, which reflecting their quality. Top-K models are then selected as HuggingGPT candidate models.


- **Task Execution**: Invoke and execute each selected model, and return the results to ChatGPT.
    - For stability and efficiency, HuggingGPT prioritizes local endpoints, running common or time-consuming models locally. Though local endpoints are faster, they cover fewer models compared to Hugging Face's inference endpoints. **If a model isn't locally deployed, HuggingGPT will use the Hugging Face endpoint.**

    - Managing resource dependencies between tasks in the task execution stage can be challenging for HuggingGPT, even with its task planning capabilities. To address this, a unique `<resource>` symbol is used to manage dependencies. HuggingGPT identifies resources generated by prerequisite tasks as `<resource>-task_id` during the task planning stage. If tasks depend on a resource generated by a task with task_id, HuggingGPT sets the symbol to the corresponding resource subfield in task arguments. During task execution, HuggingGPT dynamically substitutes the symbol with the actual resource.


        For example, as shown in the following picture, the task 3 depends on task 1(id: 1) and task 2(id: 1). So in the task-planning phase, the args of task 3 is temporarily marked with: `{"text": "<resource>-0", "image": "<resource>-1"}`. Once the task 1 and task 2 are finished, the <resource>-0 and <resource>-1 will be sustituted with "a young boy is riding a bike with a basket" and the pose.

    ![](/paper_bites/hugginggpt_1.png#center)

• **Response Generation**: Finally, ChatGPT will integrate the prediction of all models, and generate answers for users.

The details of the prompth design in HuggingGPT is shown in this table: 
![](/paper_bites/hugginggpt_2.png#center)


## Limitations of HuggingGPT
- **efficientcy**: HuggingGPT's interactions with the large language model during task planning, model selection, and response generation increase response latency, degrading user experience.
- **maximum context length**: there is limited maximum number of tokens to be accepted
- **system stability**:
    - the rebellion occurs during inferece of LLM because it occasionally fail and the output format may defy expectations
    - uncontrollable state of the expert model hosted on HuggingFace's inference endpoint which may be affected by netwirk latency or service state


[^1]: Yongliang Shen, et al. "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace." (2023). 