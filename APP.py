from flask import send_file
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import time
import json
import requests
import threading
from flask import Flask,request, jsonify, copy_current_request_context
import pickle

app = Flask(__name__)
load_dotenv()

executor = ThreadPoolExecutor()

def DeploymentTrick():
    while True:
        requests.get('https://asu-copilot.onrender.com/')
        time.sleep(3*60)
        
executor.submit(DeploymentTrick)


token = os.environ.get("WHATSAPP_TOKEN")
prepared=False

def prepare():
    global prepared
    print("Preparing...")
    global raw_text
    global text_chunks
    global vectorstore
    
    if not os.path.exists('chains'):
        os.makedirs('chains')
    with open ("bylaw2018.txt", "r") as myfile:
        raw_text = '\n'.join(myfile.readlines())
        text_chunks = get_text_chunks()
        # Create vector store
        vectorstore = get_vectorstore()
    prepared=True

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks():
    global raw_text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore():
    global text_chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    pickle.dump(vectorstore, open("vectorstore.pkl", "wb"))
    return pickle.load(open("vectorstore.pkl", "rb"))

def get_conversation_chain():

    llm = ChatOpenAI()
    global vectorstore
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain
i=0


def answer_question(prompt,phone_number_id):
    my_chain = None

    try:
        print('tryng to load',f'chains/chain_{phone_number_id}.pkl')
        my_chain=pickle.load(open(f"chains/chain_{phone_number_id}.pkl", "rb"))
        print('loaded')
    except:
        print('failed to load',f'chains/chain_{phone_number_id}.pkl','generating')
        my_chain=get_conversation_chain()
        print('generated')
        
    try:
        print('answering')
        response = my_chain({'question': prompt})
        print('dumping')
        pickle.dump(my_chain, open(f"chains/chain_{phone_number_id}.pkl", "wb"))
        print('dumped')
    except Exception as e:
        print('error',e)
    
    chat_history = response['chat_history']
    print(chat_history)
    return chat_history[-1].content




@app.route('/23413231',methods=['GET'])
def hellop():
    prepare()


@app.route('/download34234')
def downloadFile ():
    prepare()
    #For windows you need to use drive name [ex: F:/Example.pdf]
    return send_file("vectorstore.pkl", as_attachment=True)



@app.route('/',methods=['GET'])
def hello():
    return 'Hello, World!'

def post_request(body):
    global prepared
    if not prepared:
        try:
            prepare()
        except Exception as e:
            print('prep err: ',e)
    try:
        if "object" in body:
            if (
                "entry" in body
                and body["entry"][0]["changes"]
                and body["entry"][0]["changes"][0]["value"]["messages"]
                and body["entry"][0]["changes"][0]["value"]["messages"][0]
            ):
                try:
                    phone_number_id = body["entry"][0]["changes"][0]["value"]["metadata"]["phone_number_id"]
                    from_number =    body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
                    msg_body =       body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
                    headers = {"Content-Type": "application/json"}
                    url = f"https://graph.facebook.com/v12.0/{phone_number_id}/messages?access_token={token}"
                    print('Q:',msg_body)
                    reply = answer_question(msg_body,from_number)
                    print('R:',reply)
                    data = {
                        "messaging_product": "whatsapp",
                        "to": from_number,
                        "text": {"body": reply},
                    }
               
                    res= requests.post(url, json=data, headers=headers)
              
            
                except Exception as e:
                    print('no res2',e)

    except Exception as e:
        print('no res3',e)
# Accepts POST requests at /webhook endpoint
@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_json()
    try:
        future = executor.submit(post_request, body)
    except Exception as e:
        error_message = {"error": str(e)}
        return jsonify(error_message), 200
    return json.dumps({'success':True,'message':'started execution'}), 200, {'ContentType':'application/json'} 


# Accepts GET requests at the /webhook endpoint. You need this URL to setup webhook initially.
# info on verification request payload: https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests 
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    UPDATE YOUR VERIFY TOKEN
    This will be the Verify Token value when you set up webhook
    """
    print('GETTING')
    verify_token = os.environ.get("VERIFY_TOKEN")

    # Parse params from the webhook verification request
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == "subscribe" and token == verify_token:
            # Respond with 200 OK and challenge token from the request
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            return "Forbidden", 403
    print('not')
    return "Not Found", 404


""" if __name__=="__main__":
    prepare()
    i=0
    while(True):
        i+=1
        answer_question(input("ask: "),i%2)

"""
if __name__=="__main__":
    app.run()
