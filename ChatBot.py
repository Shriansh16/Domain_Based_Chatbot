import os
import streamlit as st
from streamlit_chat import message
import uuid
import modal

os.environ["MODAL_TOKEN_ID"] = st.secrets["MODAL_TOKEN_ID"]
#os.environ["MODAL_TOKEN_ID"] =""
os.environ["MODAL_TOKEN_SECRET"] = st.secrets["MODAL_TOKEN_SECRET"]
#os.environ["MODAL_TOKEN_SECRET"] = ""

# Verify authentication by creating a simple Modal client
try:
    with modal.Client() as client:
        print("Authentication successful!")
except Exception as e:
    print(f"Authentication failed: {e}")


Companion = modal.Cls.lookup("domain_based_chatbot", "Companion")
pricer = Companion()
#api_key1=""
# Streamlit setup  


if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! I'm Maria, I can assist you with queries related to Healthcare, Insurance, Finance, or Retail. How can I help you today?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []


link='icon.jpg'

response_container = st.container()
# Container for text box
text_container = st.container()



with text_container:
    user_query =st.chat_input("Let’s chat! Type your message here...")
    prompt = f"""Answer the Question
    Question: {user_query}
    Answer:
    """

    if user_query:
        with st.spinner("typing..."):
            response = pricer.query.remote(text=user_query, 
                          session_id=st.session_state['session_id'])


        
        # Append the new query and response to the session state  
        st.session_state.requests.append(user_query)
        st.session_state.responses.append(response)
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True
)


# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            with st.chat_message('Momos', avatar=link):
                st.write(st.session_state['responses'][i])
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')