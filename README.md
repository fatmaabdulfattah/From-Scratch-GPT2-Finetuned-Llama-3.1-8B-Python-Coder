# From-Scratch GPT-2 to Fine-Tuned Llama

This repository documents my deep dive into the architecture and training of Large Language Models. The project is a practical implementation journey inspired by the seminal paper ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and guided by Dr. Rajat [YouTube series by Vizuara](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu).

## 🚀 Project Overview

The project is structured into three core phases, each a significant milestone:

1.  **GPT-2 from Scratch:** Built and pre-trained a GPT-2 (124M parameter) model from scratch using PyTorch. This involved implementing the transformer decoder architecture, crafting the data pipeline, and managing the training loop.
2.  **Efficient Fine-Tuning:** Leveraged **Unsloth** to rapidly fine-tune a **Meta-Llama-3.1-8B** model on a custom dataset of Python code. This demonstrates the power of modern parameter-efficient fine-tuning (PEFT) techniques like LoRA.
3.  **Interactive Demo:** Developed a Streamlit web application to showcase the capabilities of the fine-tuned model, allowing for interactive Python code generation and completion.

## 🌐 Phase 1: Pre-Training GPT-2 on Tiny Stories Dataset

1. Imported Libraries and Modules like **TransformerBlockClass** which has the implementation from scratch of all classes and **All_Functions** which has functions of training and generate code and evaluation model.
2. Load Tiny Stories Dataset from Kaggle it has over 2M rows so I sampled 5k rows to train on them.
3. Preprocessing the text by removing special characters and normalizing whitespace and and Generates a word cloud to visualize the most frequent words in the stories.
4. Tokenized the text using **tiktoken** and Defined custom Dataset and DataLoader classes for efficient batching.
5. Trained the model by custom GPT-2 architecture and using the **train_model** function with  functions for generating text and evaluating model performance from All_Functions module.
7. Visualizing the training and validation loss curves during model training to monitor model performance.



## ⚡ Phase 2: Fine-Tuning with Unsloth 

1. Installed required dependencies like Unsloth and Datasets to enable efficient fine-tuning and data handling.
2. Loaded the base model **meta-llama-3.1-8b** with applying LoRA and the custom dataset for instruction tuning.
3. Preprocessed and tokenized the dataset, formatting it into instruction–response pairs suitable for the model’s sequence length.
4. Fine-tuned the model using Unsloth’s training pipeline, monitoring training metrics.
5. Saved the fine-tuned model and tokenizer locally and pushed to Hugging Face for later usage and deployment .
6. Performed testing and evaluation by loading the saved model and running sample prompts to verify performance.

## 🌐 Phase 3: Live Interactive Demo (Colab Deployment)

This is where the project comes alive! I deployed the fine-tuned model as a interactive web application directly from a Google Colab notebook.

**Key Features of the App:**
*   **Clean UI:** Built with Streamlit, featuring a user-friendly interface.
*   **Parameter Control:** Users can adjust key generation parameters like Temperature, Max Length, and Top-k to control the creativity and length of the output.
*   **Real-time Generation:** The app sends prompts to the loaded model and streams the generated Python code back in real-time.
*   **One-Click Deployment:** The entire setup, including model loading and public tunneling, is automated within a single Colab notebook using `pyngrok`.

**Tech Stack for Deployment:** `Streamlit`, `PyNgrok`, `Google Colab`, `Hugging Face` `transformers` & `peft`

### 🚀 How to Run the Demo Yourself (Easy!)

The beauty of this setup is that anyone can run a live instance of this app in under 5 minutes, for free, without any local setup.

1.  **Open the Colab Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/From-Scratch-GPT2-Finetuned-Llama/blob/main/Phase_3_Interactive_Demo/Live_Colab_Demo.ipynb)
2.  **Run the first cell:** It will install all dependencies.
3.  **Add your Ngrok auth token:** (Required for creating a public URL). [Get a free token here.](https://dashboard.ngrok.com/get-started/your-authtoken)
    *   In Colab, click the **key icon** (`Secrets`) on the left toolbar.
    *   Add a new secret named `NGROK_TOKEN` and paste your token.
4.  **Run the rest of the cells:** The second cell will start the server and print a public URL (e.g., `https://abc123.ngrok.io`).
5.  **Click the URL!** Your personal instance of the Llama Code Generator is now live and ready to use.

**👉 [Watch the Video Demo](./Phase_3_Interactive_Demo/demo_recording.mp4)**
*(A direct link to a screen recording of the app working)*