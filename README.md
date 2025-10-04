# 👕 Customer Feedback Analysis Engine

An interactive web application built with Streamlit to perform sentiment analysis on customer reviews. This tool leverages a pre-trained Transformer model from Hugging Face to classify feedback as positive or negative, providing businesses with immediate insights into customer satisfaction.

## 🚀 Live Demo

You can view and interact with the live application here: **[Your Streamlit App URL]**

## ✨ Features

- **Sentiment Analysis:** Automatically classifies review text into **Positive** or **Negative** categories using a BERT-based model.
- **Interactive Dashboard:** Visualizes the sentiment distribution with an interactive pie chart.
- **Dynamic Filtering:** Allows users to filter reviews based on their predicted sentiment in real-time.
- **Data Display:** Shows the raw review text alongside its predicted sentiment and confidence score.

## 🛠️ Technologies Used

- **Backend:** Python
- **Web Framework:** Streamlit
- **NLP/ML:** Hugging Face Transformers, NLTK
- **Data Manipulation:** Pandas
- **Charting:** Plotly

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/shrutimaurya11/customer-feedback-analyzer.git](https://github.com/shrutimaurya11/customer-feedback-analyzer.git)
    cd customer-feedback-analyzer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
