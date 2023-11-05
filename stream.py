import streamlit as st

tabs = st.sidebar.radio("Select Tab", ["Drug-Drug Interaction", "Medicine Reminder","AI Assistant"])

# Main app content
st.markdown('<h1 style = "color : #0079B1;text-decoration: underline; font-size : 46px; text-align: center;font-family: Courier New"> MedCompanion</h1>', unsafe_allow_html = True)

if tabs == "AI Assistant":
    def ques(prompt):
        from langchain import PromptTemplate
        from langchain.chains import RetrievalQA
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Pinecone
        import pinecone
        import tempfile
        from langchain.document_loaders.csv_loader import CSVLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.prompts import PromptTemplate
        from langchain.llms import CTransformers
        from langchain.chat_models import ChatOpenAI
        from transformers import AutoTokenizer
        import os
        import pandas as pd

        PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','2693584c-f620-496b-b7db-40232853ebdd')
        PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

        index_name = "medbot"

        path='chatbot.csv'

        loader = CSVLoader(file_path=path)
        data = loader.load()

        document_texts = [doc.page_content for doc in data]

        input_text = "\n".join(document_texts)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        tokenized_text = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        input_ids = tokenized_text["input_ids"]
        input_text = tokenizer.decode(input_ids[0])

        # Assuming 'documents' is a list of strings from the text file

        # Initialize a list to store the split documents
        split_docs = []

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

        # Split each line/document into smaller chunks
        for document in document_texts:
            split_chunks = text_splitter.split_text(input_text)
            split_docs.extend(split_chunks)

        # Now, 'split_docs' contains the smaller text chunks

        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        pinecone.init(api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV)

        docsearch=Pinecone.from_existing_index(index_name, embeddings)
        
        prompt_template="""
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that I don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        chain_type_kwargs={"prompt": PROMPT}

        llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens':512,
                                'temperature':0.8})

        qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 1}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

        
        result=qa({"query": prompt})
        return result

    import streamlit as st

    st.markdown('<h1 style="font-size: 31px;">Your personal AI Assistant</h1>', unsafe_allow_html = True)

    st.error('Please note that this chatbot provides general drug recommendations and should not replace professional medical advice.')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("enter your message here")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role":"user","content":prompt})

        response = ques(prompt)

        result = response.get("result", "No response available.")

        with st.chat_message("assistant"):
            st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})

elif tabs == "Medicine Reminder":
    import streamlit as st
    from plyer import notification
    import time
    from datetime import datetime, time as dt_time

    st.markdown('<h1 style="font-size: 31px;">Medicine Reminder</h1>', unsafe_allow_html = True)

    # Create a dictionary to store medicine details (name and time)
    medicine_data = {}

    # Function to send notifications
    def send_notification(medicine_name, scheduled_time):
        current_time = datetime.now().time()
        
        if current_time >= scheduled_time:
            st.error(f"Time for {medicine_name}: {scheduled_time.strftime('%H:%M')} has already passed.")
            return

        while True:
            current_time = datetime.now().time()
            if current_time >= scheduled_time:
                st.success(f"Time to take {medicine_name}: {scheduled_time.strftime('%H:%M')}")
                notification_title = "Medicine Reminder"
                notification_message = f"It's time to take your {medicine_name}"
                notification.notify(
                    title=notification_title,
                    message=notification_message,
                    app_name="Medicine Reminder App",
                )
                break

    # Streamlit UI
    medicine_name = st.text_input("Enter the name of the medicine:")
    scheduled_time = st.time_input("Enter the time to take the medicine:")

    if st.button("Set Reminder"):
        if medicine_name and scheduled_time:
            medicine_data[medicine_name] = scheduled_time
            st.success(f"Reminder set for {medicine_name} at {scheduled_time.strftime('%H:%M')}")

    # Display current reminders
    if medicine_data:
        st.subheader("Current Reminders:")
        for name, time in medicine_data.items():
            st.write(f"{name} at {time.strftime('%H:%M')}")

    # Send reminders
    for name, time in medicine_data.items():
        send_notification(name, time)

elif tabs == "Drug-Drug Interaction":
    import streamlit as st
    import pandas as pd

    # Load your dataset
    data = pd.read_csv("ddi_dataset.csv")  # Replace with the actual dataset file path

    # Create a Streamlit app
    st.markdown('<h1 style="font-size: 31px;">Drug Interaction Checker</h1>', unsafe_allow_html = True)

    # Input fields for two drug names
    drug1 = st.text_input("Enter Drug 1:")
    drug2 = st.text_input("Enter Drug 2:")

    # Button to check interaction
    if st.button("Check Interaction"):
        interaction = data[(data["Drug_1"] == drug1) & (data["Drug_2"] == drug2)]
        
        if not interaction.empty:
            severity = interaction["Severity"].values[0]
            description = interaction["Description"].values[0]
            
            st.success("Interaction Found!")
            st.write(f"Severity: {severity}")
            st.write(f"Description: {description}")
            if severity=='Minor':
                st.image('low_risk.png',width=200,caption='low risk')
            elif severity=='Moderate':
                st.image('moderate_risk.png',width=200,caption='moderate risk')
            elif severity=='Major':
                st.image('high_risk.png',width=200,caption='high risk')
        else:
            st.warning("No interaction found for the given drugs.")
