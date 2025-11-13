# **`Research Plan: A Comparative Analysis of White-Box vs. Black-Box Knowledge Distillation in Language Models`**

## 

## **`Executive Summary`**

This document outlines a research plan to investigate a critical, unaddressed gap in knowledge distillation (KD) for large language models (LLMs). While KD is widely used to compress large "teacher" models into smaller "student" models, the community lacks a clear, empirical benchmark on the *comparative value* of different knowledge sources. This experiment will directly compare "black-box" distillation (using only final-answer logits) against "white-box" methods (using internal hidden states and attention maps). We will use meta-llama/Meta-Llama-3-8B as the teacher and TinyLlama/TinyLlama-1.1B-Chat-v1.0 as the student, connected via custom architectural adapters. The experiment will leverage a 28-GPU Ray cluster to run 28 concurrent trials (4 experimental groups x 7 random seeds), ensuring statistical robustness. The expected outcome is a foundational benchmark that provides a clear, task-dependent answer to *how* open-source LLMs should be distilled, a highly publishable result for a top-tier NLP/ML conference.

## **1\. Introduction & Research Gap**

### **1.1. The Problem**

Large Language Models (LLMs) like the Llama-3 8B series offer powerful capabilities, but their size and computational cost make them unsuitable for many on-device or low-resource applications. Knowledge Distillation (KD) is the primary technique for "compressing" these models. It involves training a smaller "student" model (e.g., TinyLlama 1.1B) to mimic the behavior of a large, pre-trained "teacher" model.

### **1.2. The Known Methods**

Distillation techniques fall into two broad categories:

1. **Black-Box Distillation:** The student only has access to the teacher's final outputs (the logits, or "answer probabilities"). This is the only method possible when distilling from closed-source APIs (like GPT-4).  
2. **White-Box Distillation:** With open-source models (like Llama 3), the student can access the teacher's internal "thoughts," including its intermediate **hidden states** and **attention maps**.

### **1.3. The Critical Research Gap**

The literature is fragmented. Numerous papers have proposed individual methods for distilling from hidden states *or* attention maps, typically comparing their single method to a black-box baseline.

However, there is **no clear, foundational benchmark** that systematically compares these white-box signals *against each other* on modern, causal (generative) language models. We do not know:

* Is it better to align hidden states or attention maps?  
* Do these complex white-box signals offer any real benefit over simple black-box distillation on tasks like **reasoning** or **math**?  
* Does the optimal signal change depending on the task (e.g., NLU vs. Reasoning)?

This experiment is designed to fill that gap.

## **2\. Proposed Experiment: "The White-Box Benchmark"**

### **2.1. Research Question**

Given a modern teacher-student pair, what is the comparative value of distilling from:  
(a) Final Logits (Black-Box)  
(b) Final Logits \+ Final Layer Hidden States (White-Box H)  
(c) Final Logits \+ Final Layer Attention Maps (White-Box A)  
(d) All of the above (White-Box H+A)

### **2.2. Hypothesis**

We hypothesize that for simple Natural Language Understanding (NLU) tasks (e.g., sentiment analysis), all methods will perform similarly. However, for complex **reasoning** and **math** tasks, the white-box methods (b, c, d) will significantly outperform the black-box baseline (a) by successfully transferring more of the teacher's internal "thought process."

### **2.3. Experimental Groups**

We will leverage the 28-GPU cluster to run 4 experimental groups, with 7 random seeds for each group to ensure statistical robustness (4 groups \* 7 seeds \= 28 total trials).

| Group | Experiment | Distillation Signals | Purpose |
| :---- | :---- | :---- | :---- |
| **Group 1 (Baseline)** | Black-Box | L\_Task \+ L\_KD (Logits) | The standard method. |
| **Group 2** | White-Box (Hidden) | L\_Task \+ L\_KD \+ L\_Align\_Hidden | Does aligning "thoughts" help? |
| **Group 3** | White-Box (Attention) | L\_Task \+ L\_KD \+ L\_Align\_Attn | Does aligning "focus" help? |
| **Group 4** | White-Box (Combined) | L\_Task \+ L\_KD \+ L\_Align\_Hidden \+ L\_Align\_Attn | Does aligning everything help, or is it just noise? |

### **2.4. Evaluation Datasets**

A diverse set of tasks is crucial to test our hypothesis:

* **NLU:** **SST-2** (from the GLUE benchmark) for sentiment analysis.  
* **Reasoning:** A subset of **MMLU** (e.g., 5-shot high school topics).  
* **Math:** **GSM8K** (grade-school math word problems).

## **3\. Experimental Architecture**

### **3.1. Models**

* **Teacher (Frozen):** meta-llama/Meta-Llama-3-8B  
  * **Hidden Dimension (d\_teacher):** 4096  
  * **Attention Heads:** 32  
* **Student (Trainable):** TinyLlama/TinyLlama-1.1B-Chat-v1.0  
  * **Hidden Dimension (d\_student):** 2048  
  * **Attention Heads:** 32

### **3.2. Architectural "Glue" (New Trainable Layers)**

We will create a new PyTorch model class (DistillationStudent) that wraps TinyLlama and adds the following "adapter" to align the models.

* **For Hidden State Alignment:** The dimensions do not match (4096 vs 2048). We will add a new, trainable linear layer to project the student's hidden state into the teacher's space.  
  * **Module:** hidden\_state\_projector \= torch.nn.Linear(in\_features=d\_student, out\_features=d\_teacher)  
  * **Python:** projected\_student\_hidden\_state \= self.hidden\_state\_projector(student\_hidden\_state)  
* **For Attention Alignment:** The head counts match (32 vs 32). This is perfect. We can directly extract the final layer attention matrices from both models (\[batch, head, seq\_len, seq\_len\]) and compare them with no adapter needed.

### **3.3. Loss Functions**

The total loss for each experimental group will be a weighted sum:

$L\_{total} \= \\alpha \\cdot L\_{task} \+ \\beta \\cdot L\_{KD} \+ \\gamma\_1 \\cdot L\_{align\\\_hidden} \+ \\gamma\_2 \\cdot L\_{align\\\_attn}$

* $L\_{task}$ **(Task Loss):** Standard Cross-Entropy loss on the ground-truth answer. (Ensures the student is still correct).  
* $L\_{KD}$ **(Black-Box Loss):** Kullback-Leibler (KL) Divergence loss between the teacher's soft teacher\_logits and the student's logits. (The "copy the answer" loss).  
* $L\_{align\\\_hidden}$ **(White-Box H Loss):** Mean Squared Error (MSE) loss between projected\_student\_hidden\_state and teacher\_hidden\_state. (The "copy the thoughts" loss).  
* $L\_{align\\\_attn}$ **(White-Box A Loss):** Mean Squared Error (MSE) loss between student\_attention\_map and teacher\_attention\_map. (The "copy the focus" loss).

The config for each experimental group will determine which gamma values are set to zero.

## **4\. Implementation Plan (using Ray)**

This plan is designed for high-throughput, offline distillation to maximize cluster efficiency.

**Step 1: Environment Setup**

* Create a virtual environment on all nodes with torch, transformers, datasets, accelerate, and ray\[data,train,tune\].  
* Initialize Ray to connect to the cluster (ray.init()).

**Step 2: Dataset Preparation**

* Load all datasets (SST-2, MMLU, GSM8K) using the datasets library.  
* Pre-process all of them into a single, unified {"prompt": "...", "answer": "..."} format.

**Step 3: OFFLINE Teacher Data Generation (Critical Efficiency Step)**

* We will **not** run the 8B teacher model live during student training. This is "online" distillation and is extremely slow.  
* We will run the teacher *once* and save its knowledge.  
* **Action:** Write a script that uses ray.data to load the prepared datasets.  
* **Parallelism:** Use ray.data.map\_batches() to apply a @ray.remote(num\_gpus=1) function to the data. This function will run the Llama-3 8B teacher on a batch of prompts.  
* **Extraction:** This function will extract and save all necessary data:  
  1. teacher\_logits  
  2. teacher\_hidden\_state  
  3. teacher\_attention\_map  
* **Output:** Save this new, "enriched" dataset (prompts, answers, and all teacher data) to a shared network drive in Parquet format. Your 28 GPUs will make this pre-computation step very fast.

**Step 4: Create the Ray "Trainable" Function**

* Create the main Python function, def train\_student(config):, that Ray Tune will execute.  
* This function will:  
  1. Receive its config (e.g., {"distill\_type": "hidden\_state", "seed": 3}).  
  2. Load the DistillationStudent architecture (from 3.2).  
  3. Load the pre-computed *offline* dataset (from Step 3).  
  4. Run a standard training loop.  
  5. Inside the loop, calculate the loss based on config\["distill\_type"\], enabling/disabling the L\_align terms as required.  
  6. Periodically report metrics back to Ray Tune: ray.train.report({"validation\_loss": ..., "accuracy": ...}).

**Step 5: Configure and Launch Ray Tune**

* This is the main script you will run from the head node.  
* **Define Search Space:** This defines all 28 jobs.  
  search\_space \= {  
      "distill\_type": tune.grid\_search(\[  
          "black\_box",   
          "hidden\_state",   
          "attention",   
          "combined"  
      \]),  
      "seed": tune.grid\_search(list(range(7))), \# 7 random seeds  
      "learning\_rate": 1e-4,   
      \# ... other fixed params like alpha, beta, gamma  
  }

* **Configure Tuner:**  
  from ray import tune  
  from ray.tune.tuner import Tuner, TuneConfig

  \# Tell Ray each trial needs 1 GPU  
  trainable\_with\_resources \= tune.with\_resources(  
      train\_student,  
      {"cpu": 4, "gpu": 1}   
  )

  tuner \= Tuner(  
      trainable\_with\_resources,  
      param\_space=search\_space,  
      tune\_config=TuneConfig(  
          metric="validation\_accuracy", \# Metric to optimize  
          mode="max",                   \# We want to maximize it  
      )  
  )

* **Launch:**  
  results \= tuner.fit()

**Step 6: Analyze Results**

* Ray Tune will run all 28 jobs in parallel (one per GPU).  
* When complete, load the results into a pandas DataFrame:  
  df \= results.get\_dataframe()  
* You can now easily generate the final table for your paper:  
  print(df.groupby("config/distill\_type")\["accuracy"\].mean())

## **5\. Expected Outcomes & Publishability**

This experiment is highly publishable, not because it invents a single new algorithm, but because it provides a foundational, badly-needed benchmark.

* **Expected Outcome:** A set of tables and plots clearly showing which distillation signal (distill\_type) yields the best performance on which task (SST-2, MMLU, GSM8K).  
* **Why It's Publishable:**  
  1. **Resolves Fragmentation:** It provides a direct, "apples-to-apples" comparison of fragmented white-box methods.  
  2. **Modern Architecture:** It uses modern, relevant, and popular models (Llama 3 / TinyLlama) that the community is actively using.  
  3. **Task-Specific Insights:** The key finding will likely be that "reasoning" (MMLU, GSM8K) *requires* white-box signals, while "NLU" (SST-2) does not. This is a novel and actionable insight.  
  4. **Statistical Robustness:** By using 7 seeds per experiment (28 trials total), your results will be statistically sound and highly credible, a feature many academic papers lack due to compute constraints.