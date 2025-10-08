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
      
      st.session_state.saved_text = user_input
      
      import requests
      import base64
       
      url = "https://api.github.com/repos/prajwalpatil18/Database_RAG/contents/data.txt"
      headers = {"Authorization": f"token {st.secrets['GIT_API']}"}
   
      resp = requests.get(url,headers = headers)
       
      if resp.status_code == 200:
         data = resp.json()
         sha = data["sha"]
         old_content = base64.b64decode(data["content"]).decode()
      elif resp.status_code == 404:
           # File doesn't exist → create new
         sha = None
         old_content = ""
      else:
         st.error(f"GitHub API error: {resp.status_code} {resp.text}")
         return
   
       # 2️⃣ Append new text
      updated_content = old_content + st.session_state.saved_text.strip() + "\n"
   
       # 3️⃣ Encode and PUT to GitHub
      b64_content = base64.b64encode(updated_content.encode()).decode()
      put_resp = requests.put(url, headers=headers, json={
           "message": f"Append prompt from Streamlit",
           "content": b64_content,
           "sha": sha  # omit if file is new
       })
   
      if put_resp.status_code in [200, 201]:
         st.success("Prompt appended to GitHub successfully!")
      else:
         st.error(f"Failed to update GitHub file: {put_resp.status_code} {put_resp.text}")# Save latest input
      st.rerun()  # Clear input box
    
    # At this point, the latest input is stored in:
    # st.session_state.saved_text

   
   
   
    



 
