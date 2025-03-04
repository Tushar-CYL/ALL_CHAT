import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import base64

# ---------------------------
# Import Groq and Gemini libraries
# ---------------------------
from groq import Groq
import google.generativeai as genai

# ---------------------------
# Import additional libraries for Gemini functionalities
# ---------------------------
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import fitz  # PyMuPDF for PDF files
import docx  # python-docx for DOCX files
from datetime import datetime

# ---------------------------
# Import Hugging Face InferenceClient for HF Chat
# ---------------------------
from huggingface_hub import InferenceClient

# ---------------------------
# Load environment variables and configure API keys
# ---------------------------
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# You can set your Gemini API key via environment variables as well.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyATZ0NDoTweYu3S3KQBlLBo0PljHvViK30")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Hugging Face client with your token.
# You can also load the token from an environment variable for security.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")

huggingface_client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)


# ---------------------------
# Set up Streamlit app configuration
# ---------------------------
st.set_page_config(page_title="Integrated Chat & Analysis App", page_icon="🤖", layout="wide")

#########################
# GROQ CHAT FUNCTIONALITY
#########################
def encode_image(uploaded_file):
    """Encodes an uploaded image file into a base64 string."""
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_data
def fetch_available_models():
    """Fetch available models from Groq."""
    try:
        models_response = groq_client.models.list()
        return models_response.data
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def groq_chat():
    st.title("Groq Chat with LLaMA3x")
    if "groq_messages" not in st.session_state:
        st.session_state.groq_messages = []
    if "groq_image_used" not in st.session_state:
        st.session_state.groq_image_used = False

    with st.sidebar:
        st.header("Groq Chat Options")
        available_models = fetch_available_models()
        filtered_models = [model for model in available_models if 'llama' in model.id]
        models_dict = {
            model.id: {"name": model.id, "tokens": 4000, "developer": model.owned_by}
            for model in filtered_models
        }
        if models_dict:
            model_option = st.selectbox(
                "Choose a model:",
                options=list(models_dict.keys()),
                format_func=lambda x: f"{models_dict[x]['name']} ({models_dict[x]['developer']})"
            )
        else:
            st.warning("No available models to select.")
            model_option = None

        if models_dict and model_option:
            max_tokens_range = models_dict[model_option]["tokens"]
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=200,
                max_value=max_tokens_range,
                value=max(100, int(max_tokens_range * 0.5)),
                step=256,
                help=f"Adjust the maximum number of tokens for the response. Maximum: {max_tokens_range}"
            )
        else:
            max_tokens = 200

        stream_mode = st.checkbox("Enable Streaming", value=True)
        if st.button("Clear Chat"):
            st.session_state.groq_messages = []
            st.session_state.groq_image_used = False

        base64_image = None
        uploaded_file = None
        if model_option and "vision" in model_option.lower():
            st.markdown("### Upload an Image (one per conversation)")
            if not st.session_state.groq_image_used:
                uploaded_file = st.file_uploader(
                    "Upload an image for the model to process:",
                    type=["png", "jpg", "jpeg"],
                    help="Upload an image if the model supports vision tasks.",
                    accept_multiple_files=False
                )
                if uploaded_file:
                    base64_image = encode_image(uploaded_file)
                    st.image(uploaded_file, caption="Uploaded Image")
        else:
            base64_image = None

    st.markdown("### Chat Interface")
    for message in st.session_state.groq_messages:
        avatar = "🔋" if message["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        if item["image_url"]["url"].startswith("data:image"):
                            st.image(item["image_url"]["url"], caption="Uploaded Image")
                            st.session_state.groq_image_used = True
                        else:
                            st.warning("Invalid image format.")
            else:
                st.markdown(message["content"])

    user_input = st.chat_input("Enter your message here...")
    if user_input:
        if base64_image and not st.session_state.groq_image_used:
            st.session_state.groq_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            )
            st.session_state.groq_image_used = True
        else:
            st.session_state.groq_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
            if base64_image and st.session_state.groq_image_used:
                st.image(uploaded_file, caption="Uploaded Image")
                base64_image = None

        try:
            full_response = ""
            usage_summary = ""
            if stream_mode:
                chat_completion = groq_client.chat.completions.create(
                    model=model_option,
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.groq_messages],
                    max_tokens=max_tokens,
                    stream=True
                )
                with st.chat_message("assistant", avatar="🔋"):
                    response_placeholder = st.empty()
                    for chunk in chat_completion:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response)
            else:
                chat_completion = groq_client.chat.completions.create(
                    model=model_option,
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.groq_messages],
                    max_tokens=max_tokens,
                    stream=False
                )
                response = chat_completion.choices[0].message.content
                usage_data = chat_completion.usage
                with st.chat_message("assistant", avatar="🔋"):
                    st.markdown(response)
                full_response = response
                if usage_data:
                    usage_summary = (
                        f"**Token Usage:**\n"
                        f"- Prompt Tokens: {usage_data.prompt_tokens}\n"
                        f"- Response Tokens: {usage_data.completion_tokens}\n"
                        f"- Total Tokens: {usage_data.total_tokens}\n\n"
                        f"**Timings:**\n"
                        f"- Prompt Time: {round(usage_data.prompt_time,5)} secs\n"
                        f"- Response Time: {round(usage_data.completion_time,5)} secs\n"
                        f"- Total Time: {round(usage_data.total_time,5)} secs"
                    )
            if usage_summary:
                st.sidebar.markdown("### Usage Summary")
                st.sidebar.markdown(usage_summary)
            st.session_state.groq_messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error generating response: {e}")

#########################
# GEMINI CHAT WITH DOCUMENT ANALYSIS
#########################
ANALYSIS_PROMPTS = {
    "key_points": """
    Analyze the following document and extract the top 5-7 most important key points. 
    Provide a concise summary that captures the essential information in a clear, structured manner.
    """,
    "potential_questions": """
    Based on the document content, generate 5-7 potential questions that a reader might ask. 
    Provide both the questions and brief, informative answers.
    """,
    "summary": """
    Provide a comprehensive summary of the document. 
    Capture the main ideas, key arguments, and overall context of the text in a detailed yet concise way.
    """
}

def get_gemini_response(content, prompt):
    """Generate response using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt, content])
        return response.text
    except Exception as e:
        st.error(f"Error getting response from GenAI: {e}")
        return None

def read_pdf(file):
    """Extract text from a PDF file."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def read_docx(file):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def read_uploaded_file(file):
    """Determine file type and extract text."""
    try:
        if file is not None:
            file_type = file.type
            if 'pdf' in file_type:
                return read_pdf(file)
            elif 'wordprocessingml' in file_type:
                return read_docx(file)
            else:
                st.error("Unsupported file type.")
                return None
        return None
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def gemini_chat():
    st.title("Gemini Chat with Document Analysis")
    if "gemini_messages" not in st.session_state:
        st.session_state["gemini_messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    with st.expander("Upload Document for Analysis"):
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx'],
            help="Upload a document to analyze."
        )
        if uploaded_file:
            analysis_option = st.radio(
                "Select Analysis Type",
                ["Key Points", "Potential Questions", "Summary"]
            )
            if st.button("Analyze Document"):
                file_content = read_uploaded_file(uploaded_file)
                if file_content:
                    if analysis_option == "Key Points":
                        prompt = ANALYSIS_PROMPTS["key_points"]
                    elif analysis_option == "Potential Questions":
                        prompt = ANALYSIS_PROMPTS["potential_questions"]
                    else:
                        prompt = ANALYSIS_PROMPTS["summary"]
                    with st.spinner("Analyzing document..."):
                        analysis_result = get_gemini_response(file_content, prompt)
                    st.markdown(f"### {analysis_option}")
                    st.write(analysis_result)
                    st.session_state.gemini_messages.append({
                        "role": "assistant",
                        "content": f"Document Analysis - {analysis_option}:\n\n{analysis_result}"
                    })
                else:
                    st.error("Unable to read the uploaded file.")
    
    for msg in st.session_state.gemini_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_prompt = st.chat_input("Enter your message here...")
    if user_prompt:
        st.session_state.gemini_messages.append({"role": "user", "content": user_prompt})
        st.chat_message("user").write(user_prompt)
        response = get_gemini_response(user_prompt, user_prompt)
        if response:
            st.session_state.gemini_messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            st.error("No valid response could be generated.")

#########################
# HUGGING FACE CHAT FUNCTIONALITY
#########################
def huggingface_respond_generator(message, history, system_message, max_tokens, temperature, top_p):
    """
    Generator that calls Hugging Face InferenceClient to stream responses.
    """
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    response = ""
    for message_obj in huggingface_client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message_obj.choices[0].delta.content
        response += token
        yield response

def huggingface_chat():
    st.title("Hugging Face Chat")
    if "huggingface_messages" not in st.session_state:
        st.session_state.huggingface_messages = [{"role": "assistant", "content": "How can I help you?"}]
    with st.sidebar:
        st.header("Hugging Face Chat Options")
        system_message = st.text_input("System Message", value="You are a friendly Chatbot.")
        max_tokens = st.slider("Max New Tokens", 1, 2048, 512)
        temperature = st.slider("Temperature", 0.1, 4.0, 0.7, step=0.1)
        top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95, step=0.05)

    for msg in st.session_state.huggingface_messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    
    user_input = st.chat_input("Enter your message here...")
    if user_input:
        st.session_state.huggingface_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        full_response = ""
        try:
            response_placeholder = st.empty()
            for partial in huggingface_respond_generator(
                user_input,
                st.session_state.huggingface_messages,
                system_message,
                max_tokens,
                temperature,
                top_p,
            ):
                full_response = partial
                response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
        st.session_state.huggingface_messages.append({"role": "assistant", "content": full_response})

#########################
# MAIN NAVIGATION
#########################
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose an App Mode",
    ["Groq Chat", "Gemini Chat", "Hugging Face Chat"]
)

if app_mode == "Groq Chat":
    groq_chat()
elif app_mode == "Gemini Chat":
    gemini_chat()
elif app_mode == "Hugging Face Chat":
    huggingface_chat()
