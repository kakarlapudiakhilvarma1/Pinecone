# 📄 PDF RAG Application with Gemini, Pinecone, and Streamlit

This project is a **RAG (Retrieval-Augmented Generation)** application built using **LangChain**, **Google Generative AI (Gemini)**, **Pinecone**, and **Streamlit**. It allows you to upload and process PDF documents and **ask contextual questions** based on the content.

---

## 🚀 Features

- 🔍 Loads PDFs from a directory using `PyPDFDirectoryLoader`
- 🧠 Splits and embeds text using `Google Generative AI Embeddings`
- 📚 Stores vectorized data in **Pinecone**
- 🤖 Uses **Gemini (Google LLM)** to answer questions from retrieved documents
- 🧵 Uses LangChain's `retrieval_chain` and `stuff_documents_chain`
- 🖼️ Simple UI with **Streamlit**

---

## 🗂️ Project Structure

```
pinecone
├── .env                        # API Keys
├── main.py                    # Entry point for Streamlit app
├── pdf_files/                 # Directory to place your PDF files
└── src/
    ├── config/
    │   └── settings.py        # Environment variable management
    ├── utils/
    │   ├── pdf_loader.py      # Loads and splits PDFs
    │   └── prompt_templates.py# Prompt templates used for Gemini
    ├── services/
    │   ├── pinecone_service.py# Pinecone initialization and vector store creation
    │   ├── embedding_service.py# Embedding generation with Google GenAI
    │   └── rag_service.py     # Chain creation for RAG logic
    └── ui/
        └── streamlit_ui.py    # Streamlit UI components
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## 📦 Installation

```bash
git clone https://github.com/kakarlapudiakhilvarma1/Pinecone.git
cd pinecone
conda create -p myenv python==3.10 -y
conda activate myenv/  #windows
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run main.py
```

---

## 📥 How to Use

1. Place your PDF documents in the `pdf_files/` directory.
2. Click **"Process PDF Files"** in the sidebar.
3. Ask questions in the input box on the main screen.
4. Get **short, structured, and contextually accurate responses** powered by Gemini!

---

## Example Questions
- No Connection to Unit
- Failure in Optical Interface
- BCCH Missing

---

## Output 

![image](https://github.com/user-attachments/assets/0839b7c3-a48a-4cec-994e-26cc84520393)

![image](https://github.com/user-attachments/assets/9ce04b0f-d3bd-4216-95f0-4ddccab8ded4)

![image](https://github.com/user-attachments/assets/6b4030f5-839e-4668-ac71-3d05cfbc2d37)

![image](https://github.com/user-attachments/assets/703310c8-124e-4998-8cd1-75ce2e8aa941)


## 🧠 Prompt Design

Special prompt logic is embedded to act as a **Telecom NOC Engineer** when questions relate to alarms or radio issues. The response is structured as:

1. **Response**
2. **Explanation of the issue**
3. **Recommended steps**
4. **Quality steps** (with best practices like TSDANC format, INC/CRQ history, escalation rules)

---

## 🧰 Tech Stack

| Tool            | Purpose                              |
|-----------------|--------------------------------------|
| **LangChain**   | RAG pipeline and LLM chaining        |
| **Pinecone**    | Vector store                         |
| **Google GenAI**| Embeddings + LLM (Gemini 1.5 Pro)    |
| **Streamlit**   | Web UI                               |
| **PyPDFLoader** | PDF reading                          |

---

## 📌 Notes

- Make sure your documents are in the `pdf_files/` folder before clicking **Process PDF Files**.
- You must process documents at least once before asking questions.
- This app is **customizable** for any domain – just change the prompt template!

---

## 🛡️ License

MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Google Generative AI](https://makersuite.google.com/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [Streamlit](https://streamlit.io/)

---
