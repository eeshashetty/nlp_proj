# nlp_proj

## Requirements
 1. `pip3 install -r requirements.txt`
 2. Install punkt
 ```
 import nltk
 nltk.download('punkt')
 ```
 
 ## Using the Chatbot
 1. `python3 chatbot.py`
 wait for it to load, then once the Bot gives a welcome message, you can start using it.
 (for type of questions you can ask, check out queries in data.csv)
 
 2. To see model architecture
 its printed when you run (2). If you only want to see architecture, 
 `python3 load.py`

3. In `chatbot.py` you can edit the `messages` dictionary to alter question responses
