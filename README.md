# 🧠 From-Scratch GPT-2 to Fine-Tuned Llama

This repository documents my deep dive into the architecture and training of Large Language Models. The project is a practical implementation journey inspired by the seminal paper ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and guided by the excellent [YouTube series by nanohub](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu).

## 🚀 Project Overview

The project is structured into three core phases, each a significant milestone:

1.  **GPT-2 from Scratch:** Built and pre-trained a GPT-2 (124M parameter) model from the ground up using PyTorch. This involved implementing the transformer decoder architecture, crafting the data pipeline, and managing the training loop.
2.  **Efficient Fine-Tuning:** Leveraged **Unsloth** to rapidly fine-tune a **Meta-Llama-3-8B** model on a custom dataset of Python code. This demonstrates the power of modern parameter-efficient fine-tuning (PEFT) techniques like QLoRA.
3.  **Interactive Demo:** Developed a Gradio/Streamlit web application to showcase the capabilities of the fine-tuned model, allowing for interactive Python code generation and completion.

## 📁 Repository Structure
