# English-to-Spanish Translator 🌍

🎯 **Try the live translator app here:**  
👉 [https://englishtospanish.streamlit.app/](https://englishtospanish.streamlit.app/)

This project implements a Transformer-based English-to-Spanish translator using TensorFlow and Keras. It allows users to input an English sentence and instantly receive a Spanish translation.

---

## ⚙️ How It Works

- Model architecture: Transformer encoder-decoder (4 layers, 128-dim embedding)
- Preprocessing: Keras `TextVectorization` layers
- Inference: Greedy decoding with `[start]` and `[end]` tokens
- Model and vectorizers are stored on Google Drive and downloaded at runtime using `gdown`
