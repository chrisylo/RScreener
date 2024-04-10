
import streamlit as st
import openai
from langchain_community.chat_models import ChatOpenAI
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
import pandas as pd
import tempfile
import fitz  # PyMuPDF

## PDF to text extraction function
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# API key
client = openai.OpenAI(api_key=st.secrets["key"])

# initiate chat
chat = ChatOpenAI(model = "gpt-3.5-turbo-0125", api_key=st.secrets["key"])

## web form #############################################################################
st.title("Demo")

st.markdown("**Please fill the below form :**")
with st.form(key="Form :", clear_on_submit = True):
    job_desc = st.text_input("Job Description : ")
    cand1 = st.text_input("Candidate 1 : ")
    cand2 = st.text_input("Candidate 2: ")
    cand3_file = st.file_uploader(label = "Candidate 3 - Upload PDF", type=["pdf","docx"])
    submit = st.form_submit_button(label='Submit')
    

st.subheader("About the job")
st.write("")
#st.metric(label = "Key skills required :", value = job_desc)
#st.metric(label = "Email ID :", value = email)

if submit :
    ## ChatGPT ##################################################
    ## assistant role
       
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a recruitment specialist. You will help me to perform a suitability assessment for a job based on the candidate profiles I provide to you. Do not create additional candidates, modify or add to the profile of the candidates.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

        
    ## set up chain
    chain = prompt | chat
       
    ## start chat
    ## About the job
    recruit_chat_history = ChatMessageHistory()
    recruit_chat_history.add_user_message(f'this is the job description: {job_desc}. Please response in Markdown the job title and a summary of the skills required for this job in numbered dot points and no more than 5 dot points, each point with the headline only?')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    st.markdown(response.content)
    recruit_chat_history.add_ai_message(response)
    job_req = response.content

    # requirement list
    recruit_chat_history.add_user_message('Can you give requirement 1 a name? In your response, provide only the name (and no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    job_req_1 = response.content

    recruit_chat_history.add_user_message('Can you give requirement 2 a name? In your response, provide only the name (and no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    job_req_2 = response.content

    recruit_chat_history.add_user_message('Can you give requirement 3 a name? In your response, provide only the name (and no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    job_req_3 = response.content

    recruit_chat_history.add_user_message('Can you give requirement 4 a name? In your response, provide only the name (and no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    job_req_4 = response.content

    recruit_chat_history.add_user_message('Can you give requirement 5 a name? In your response, provide only the name (and no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    job_req_5 = response.content

    st.subheader('Key requirements')
    st.write(f'{job_req_1}, {job_req_2}, {job_req_3}, {job_req_4}, {job_req_5}')

    ## Candidate 1
    recruit_chat_history.add_user_message(f'this is the profile of candidate 1: {cand1}. What is the name of this candidate? In your response, include his/her name only (no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_name = response.content
    
    recruit_chat_history.add_user_message('Base on this profile, please assess the suitability of this candidate against the job by the requirements you listed out. In your response, only include a short paragraph on your assessment')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_assessment = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 1? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_score_1 = response.content
 
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 2? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_score_2 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 3? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_score_3 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 4? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_score_4 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 5? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand1_score_5 = response.content
    
    ## Candidate 2
    recruit_chat_history.add_user_message(f'this is the profile of candidate 2: {cand2}. What is the name of this candidate? In your response, include his/her name only (no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_name = response.content
    
    recruit_chat_history.add_user_message('Base on this profile, please assess the suitability of this candidate against the job by the requirements you listed out. In your response, only include a short paragraph on your assessment')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_assessment = response.content

    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 1? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_score_1 = response.content
 
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 2? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_score_2 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 3? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_score_3 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 4? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_score_4 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 5? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand2_score_5 = response.content    

    ## Candidate 3 PDF
    # process PDF
    temp_dir = tempfile.mkdtemp()
    cand3_file_path = os.path.join(temp_dir, cand3_file.name)
    cand3 = extract_text_from_pdf(cand3_file_path)
    
    recruit_chat_history.add_user_message(f'this is the profile of candidate 3: {cand3}. What is the name of this candidate? In your response, include his/her name only (no other words or punctuation)')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_name = response.content
    
    recruit_chat_history.add_user_message('Base on this profile, please assess the suitability of this candidate against the job by the requirements you listed out. In your response, only include a short paragraph on your assessment')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_assessment = response.content

    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 1? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_score_1 = response.content
 
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 2? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_score_2 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 3? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_score_3 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 4? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_score_4 = response.content
    
    recruit_chat_history.add_user_message('in a scale of 10, how would you score the suitability of this candidate against requirement 5? In your response, provide only the score (with no words or punctuation')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    recruit_chat_history.add_ai_message(response)
    cand3_score_5 = response.content    
        
    ## Summary score table  
    scores = {
        "Requirements":[job_req_1, job_req_2, job_req_3, job_req_4, job_req_5],
        f'{cand1_name}':[cand1_score_1, cand1_score_2, cand1_score_3, cand1_score_4, cand1_score_5],
        f'{cand2_name}':[cand2_score_1, cand2_score_2, cand2_score_3, cand2_score_4, cand2_score_5],
        f'{cand3_name}':[cand3_score_1, cand3_score_2, cand3_score_3, cand3_score_4, cand3_score_5],
                }
    
    scores_table = pd.DataFrame(scores)
    
    ## Compare
    recruit_chat_history.add_user_message('who would you recommend for the job and why?')
    response = chain.invoke({"messages": recruit_chat_history.messages})
    #st.markdown(response.content)
    recruit_chat_history.add_ai_message(response)
    recom_cand = response.content
    
    ## Display results
    st.write("")
    st.subheader("Recommendation")
    st.write("")   
    st.write(recom_cand)
    st.write("")
    st.subheader("Summary scores")
    st.write("")  
    st.dataframe(scores_table)
    st.write("")  
    st.subheader(f'Suitability of {cand1_name}')
    st.write("")
    st.markdown(cand1_assessment)
    st.write("")
    st.subheader(f'Suitability of {cand2_name}')
    st.write("")
    st.markdown(cand2_assessment)
    st.write("")    
    st.subheader(f'Suitability of {cand3_name}')
    st.write("")
    st.markdown(cand3_assessment)
    st.write("")      
    

    

