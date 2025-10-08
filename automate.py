# app.py
def saving_answer():
   import streamlit as st
    
   st.title("Save Single User Input")
    
    # Initialize session state variable for single string
   if "saved_text" not in st.session_state:
      
      st.session_state.saved_text = ""
    
    # Text input box
   user_input = st.text_input("Enter some text:")
    
    # When user hits Enter
   if user_input:
      
      st.session_state.saved_text = user_input  # Save latest input
      st.rerun()  # Clear input box
    
    # At this point, the latest input is stored in:
    # st.session_state.saved_text
   
   
   
    



 
