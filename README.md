# Domain-Based Chatbot
Deployed link: https://domainbasedchatbot.streamlit.app/                                                             
(Note: It will take some time for the first query.)

## Steps Followed  

## Step 1 Domain-Specific Q&A Dataset Preparation

This step focuses on collecting, preprocessing, and organizing datasets for question-answering tasks across various domains, including:

- **Medical Q&A**
- **Insurance Q&A**
- **Finance Q&A**
- **Retail Q&A**
- **Non-related Q&A** (general questions unrelated to the domains mentioned above)

### Data Sources
The datasets were collected from trusted platforms to ensure quality and diversity:

- Hugging Face
- Kaggle

### Dataset Details
- **Categories**:
  - Medical Q&A
  - Insurance Q&A
  - Finance Q&A
  - Retail Q&A
  - Non-related Q&A
- **Dataset Size**:
  - ~5,000 question-answer pairs per category
  - **Total Dataset**: Approximately 55,000 question-answer pairs
  dataset link: https://huggingface.co/datasets/MLsheenu/domain_specific_chatbot_dataset_v2


### Storage
After preprocessing, the datasets were stored in my Hugging Face Repository for easy accessibility and future use.


## Step 2: LLM Selection and Fine-Tuning

### LLM Selection
The selected model for this project is **Mistral-7B-Instruct-v0.3**. This model was chosen for its robust instruction-following capabilities and suitability for fine-tuning on custom datasets.

### Accessing the Model
The model was accessed via the Hugging Face Hub. Hugging Face's integration made it seamless to load and fine-tune the model.

### Fine-Tuning
To adapt the model to the prepared dataset, I employed the **QLoRA (Quantized Low-Rank Adaptation)** method. This approach allowed efficient fine-tuning while maintaining the model's performance on the domain-specific data.

### Fine-Tuning Steps:
1. **Dataset Preparation**:
   - Used the preprocessed datasets from Step 1.
   - Loaded the datasets into a format compatible with the Hugging Face Trainer.
   
2. **QLoRA Method**:
   - Applied quantized low-rank adaptation for efficient fine-tuning.
   - Reduced resource usage without compromising model accuracy.
   
3. **Training**:
   - Configured the training parameters (learning rate, batch size, etc.).
   - Fine-tuned the model on the datasets across all five domains.

4. **Model Storage**:
   - Saved the fine-tuned model in my Hugging Face repository for deployment and further experimentation.


## Step 3: Framework and Deployment

### Framework Used
The **LangChain** framework was utilized for building conversational capabilities. LangChain provides powerful tools for memory management and query construction, making it an ideal choice for this project.

### Deployment
The fine-tuned LLM was deployed on **Modal**, a cloud platform that simplifies deploying machine learning models. The deployment allows interaction with the model via an API by sending a query and receiving a response.

### Code Overview
Below is an overview of the deployment code structure:

#### Infrastructure Setup
- Defined the infrastructure using **Modal**:
  - Created a shared **volume** to store models and tokenizer.
  - Used an **A100 GPU** for efficient execution.
  - Configured the environment with required libraries like `huggingface`, `torch`, `transformers`, `bitsandbytes`, and `langchain`.

#### Deployment Steps
1. **Model and Tokenizer Download**:
   - Base model: `mistralai/Mistral-7B-Instruct-v0.3`
   - Fine-tuned model: `MLsheenu/Domain_based_chatbot_mistral_v2`
   - Models and tokenizer were downloaded and stored in a shared volume.

2. **Model Setup**:
   - Loaded the base and fine-tuned models using Hugging Face's `AutoModelForCausalLM` and `PeftModel`.
   - Applied quantization using **BitsAndBytesConfig** for memory-efficient model loading.
   - Configured tokenizer with padding and truncation settings for seamless input handling.

3. **Query Handling**:
   - Used **LangChain's memory management** to maintain conversation history.
   - Constructed prompts dynamically using the conversation's recent history.
   - Generated responses using the fine-tuned model, with settings for repetition penalty, temperature, and top-p sampling.

4. **Response Cleaning**:
   - Cleaned up generated responses by removing unnecessary tokens and patterns.
   - Maintained conversation history by appending both user queries and AI responses.

5. **API Integration**:
   - Exposed the query function as an API endpoint on Modal for remote interaction.
   - Added a wake-up function for health checks.


## Step 4: Creating Web App and User Interface

### Framework Used
The web app and UI were created using **Streamlit**, a lightweight Python library designed for building interactive web applications. The UI is conversational, offering a chat-like experience for users.

### Features
- **Interactive Chat Interface**: A user-friendly chat UI to interact with the fine-tuned AI model.
- **Session-Based Conversation**: Each user session is uniquely identified to maintain context across messages.
- **Customizable Avatar**: A friendly avatar named *Momos* is displayed alongside the AI's responses.
- **Live Typing Indicator**: Displays a "typing..." spinner while generating responses for better user experience.

### Code Overview
Below is a breakdown of the components and functionality:

#### Authentication
- **Environment Variables**: Uses `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for authenticating with Modal.
- **Verification**: Ensures authentication success before connecting to the deployed AI model.

#### Chat Functionality
- **Session Management**:
  - Each user session is assigned a unique identifier (`session_id`) using Python's `uuid` library.
  - Session state stores user queries (`requests`) and AI responses (`responses`).

- **Message Handling**:
  - Users input their queries in the chat input box.
  - AI responses are fetched by calling the deployed Modal API (`pricer.query.remote()`).

- **Response Rendering**:
  - Chat history is displayed sequentially with alternating user queries and AI responses.
  - AI responses are visually distinct with an avatar for better UX.


## Step 5: Deployment on Streamlit Cloud

The web app, developed using Streamlit, was deployed on **Streamlit Cloud**, making it accessible online for users. Streamlit Cloud provides a seamless hosting solution for Streamlit applications with minimal configuration.



