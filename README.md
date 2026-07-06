# Simple GK Answering System using RNN

This project implements a simple General Knowledge (GK) question-answering system using a Recurrent Neural Network (RNN) in PyTorch. The model is trained on a GK question-answer dataset and learns to predict an answer token based on the input question.

The project demonstrates the basic pipeline of Natural Language Processing (NLP), including text preprocessing, tokenization, vocabulary creation, converting text into numerical indices, building a custom PyTorch dataset, training an RNN model, and generating predictions.

---

## Project Overview

The aim of this project is to build a beginner-friendly question-answering system using a simple RNN architecture. The system takes a general knowledge question as input and predicts an answer based on the learned patterns from the dataset.

This project is not designed as a large-scale chatbot or advanced language model. Instead, it focuses on understanding the fundamental concepts behind sequence processing and text-based prediction using PyTorch.

---

## Dataset

The dataset used in this project contains general knowledge questions and their corresponding answers.

The dataset includes two main columns:

| Column     | Description                     |
| ---------- | ------------------------------- |
| `question` | General knowledge question      |
| `answer`   | Correct answer for the question |

Example:

| Question                                 | Answer |
| ---------------------------------------- | ------ |
| What is the capital of France?           | Paris  |
| Which planet is known as the red planet? | Mars   |
| What is H2O commonly called?             | Water  |

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Regular Expressions

---

## Project Workflow

The project follows these main steps:

1. Load the GK question-answer dataset
2. Clean and tokenize the text data
3. Build a vocabulary from the dataset
4. Convert questions and answers into numerical indices
5. Create a custom PyTorch Dataset
6. Load data using DataLoader
7. Build a simple RNN model
8. Train the model using CrossEntropyLoss
9. Create a prediction function
10. Test the model on sample GK questions

---

## Text Preprocessing

Before feeding the text into the model, the questions and answers are cleaned and tokenized.

The preprocessing steps include:

* Converting text to lowercase
* Removing quotation marks
* Removing punctuation
* Removing extra spaces
* Splitting text into individual tokens

Example:

```python
"What is the capital of France?"
```

After tokenization:

```python
['what', 'is', 'the', 'capital', 'of', 'france']
```

---

## Vocabulary Creation

A vocabulary is created from all words appearing in the questions and answers. Each unique word is assigned a numerical index.

An `<UNK>` token is also used to handle unknown words that are not present in the vocabulary.

Example:

```python
vocab = {
    '<UNK>': 0,
    'what': 1,
    'is': 2,
    'the': 3,
    'capital': 4,
    'france': 5,
    'paris': 6
}
```

---

## Text to Indices

Since neural networks cannot directly understand text, each question and answer is converted into a sequence of numerical indices using the vocabulary.

Example:

```python
"What is the capital of France?"
```

May be converted into:

```python
[1, 2, 3, 4, 5]
```

---

## Model Architecture

The model is a simple RNN-based neural network implemented using PyTorch.

### Architecture Components

| Layer                 | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| Embedding Layer       | Converts word indices into dense vector representations |
| RNN Layer             | Processes the sequence of word embeddings               |
| Fully Connected Layer | Produces output scores for words in the vocabulary      |

---

## RNN Model Structure

```python
class simpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, question):
        embedded = self.embedding(question)
        _, final = self.rnn(embedded)

        return self.fc(final.squeeze(0))
```

---

## How the Model Works

1. The input question is first converted into word indices.
2. The embedding layer converts each word index into a dense vector.
3. The RNN processes the sequence of word vectors.
4. The final hidden state of the RNN represents the meaning of the question.
5. The fully connected layer predicts the most likely answer token.
6. The predicted index is converted back into a word using the index-to-word dictionary.

---

## Training Process

The model is trained using:

* Loss function: `CrossEntropyLoss`
* Optimizer: `Adam`
* Learning rate: `0.0001`
* Epochs: `50`

During training, the model receives a question as input and learns to predict the first token of the corresponding answer.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 50
```

---

## Prediction

After training, the model can predict an answer token for a given question.

Example test questions:

```python
tests = [
    "What is the capital of France?",
    "What is H2O commonly called?",
    "Which planet is known as the red planet?",
    "Which bird cannot fly?",
    "What is the capital of Japan?"
]
```

The model returns the predicted answer word for each question.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/simple-gk-answering-system-using-rnn.git
```

### 2. Navigate to the project folder

```bash
cd simple-gk-answering-system-using-rnn
```

### 3. Install required libraries

```bash
pip install pandas torch numpy
```

### 4. Open the notebook

```bash
jupyter notebook simple_gk_answering_system_using_rnn.ipynb
```

### 5. Run all cells

Run the notebook cells step by step to:

* Load the dataset
* Build the vocabulary
* Create the dataset and dataloader
* Train the RNN model
* Test predictions

---

## Suggested Repository Name

```text
simple-gk-answering-system-using-rnn
```

---

## Suggested Main File Name

```text
simple_gk_answering_system_using_rnn.ipynb
```

---

## Key Learning Outcomes

Through this project, I learned:

* How to preprocess text data for NLP tasks
* How to tokenize sentences
* How to build a vocabulary from text
* How to convert words into numerical indices
* How to create a custom PyTorch Dataset
* How to use DataLoader for batching
* How to build a simple RNN model in PyTorch
* How embedding layers represent words as vectors
* How RNNs process sequential data
* How to train a text-based prediction model

---

## Limitations

This is a basic RNN-based question-answering system, so it has some limitations:

* It predicts only one answer token instead of a full sentence
* It depends heavily on the vocabulary created from the training dataset
* It may not perform well on completely unseen questions
* It does not understand context like advanced transformer-based models
* It is mainly designed for educational and learning purposes

---

## Future Improvements

Possible improvements include:

* Predicting full answer sequences instead of only one word
* Using LSTM or GRU instead of a basic RNN
* Adding padding for batch training with larger batch sizes
* Using a larger and more diverse GK dataset
* Applying attention mechanisms
* Building a sequence-to-sequence question-answering model
* Comparing RNN, LSTM, GRU, and Transformer-based approaches

---

## Project Type

This project is suitable for:

* Beginner NLP practice
* PyTorch learning
* RNN implementation practice
* Academic coursework
* GitHub portfolio project
* Machine learning project demonstration

---

## Author

**Abdur Rahman**

---

## Conclusion

This project demonstrates how a simple RNN can be used for a basic question-answering task. It covers the complete NLP workflow from raw text preprocessing to model training and prediction. Although the model is simple, it provides a strong foundation for understanding more advanced NLP architectures such as LSTM, GRU, sequence-to-sequence models, and transformer-based language models.
