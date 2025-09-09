# Import necessary libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os

# Set the Groq API key from Streamlit secrets
# Make sure to set a secret named 'GROQ_API_KEY' in your Streamlit Cloud app
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

# Create a prompt template for generating tweets
tweet_template = "Generate {number} tweets about the following topic: {topic}. Each tweet should be engaging and concise."

tweet_prompt = PromptTemplate(
    template=tweet_template,
    input_variables=['number', 'topic']
)

# Initialize the Groq model
# Using Llama3 8b model, which is fast and efficient on Groq
groq_model = ChatGroq(model_name="llama-3.3-70b-versatile")

# Create the LLM chain using LangChain Expression Language (LCEL)
tweet_chain = tweet_prompt | groq_model

# --- Streamlit App ---

st.set_page_config(page_title="Tweet Generator", layout="centered")
st.title("Tweet Generator with Groq")
st.subheader("Generate tweets instantly using the power of Groq's LPU")

# Get user input for the topic
topic = st.text_input("Enter the topic for the tweets:", placeholder="e.g., The future of AI")

# Get user input for the number of tweets
number = st.number_input(
    "Number of tweets to generate:",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)

# Generate tweets when the button is clicked
if st.button("Generate Tweets"):
    if topic:
        with st.spinner("Generating tweets..."):
            try:
                # Invoke the chain with the user's input
                response = tweet_chain.invoke({"number": number, "topic": topic})
                
                # Display the generated tweets
                st.success("Tweets generated successfully!")
                st.write(response.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic to generate tweets.")
