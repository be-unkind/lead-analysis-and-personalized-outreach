import os
import json
import torch
import pickle

import chromadb
import uuid

from configs import db_config
from configs import models_config

from langdetect import detect

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import utils

from groq import Groq

from dotenv import load_dotenv
load_dotenv()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DB_PATH = os.path.join(os.getcwd(), db_config.DB_NAME)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GROQ_CLIENT = Groq(
    api_key=GROQ_API_KEY,
)

def load_candidate_files(folder_path: str) -> list[dict]:
    '''
    Read files with the descriptions of candidates
    '''
    # print('start load data')
    candidates = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                number = int(filename.split('.')[0])
                candidates.append({'file_number': number, 'data': data})

    return candidates

def generate_summary(candidate_data):
    ''' 
    Generate candidate profile summary using the available information
    '''
    # print('start generating summary')
    candidate_info = utils.exctract_info_for_summary(candidate_data)

    # input_text = f'''
    #     Generate candidate profile summary in English:
    #     {candidate_info}

    #     Return only the translated text block without any introductory phrases and explanations!
    # '''

    chat_completion = GROQ_CLIENT.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Generate candidate profile summary in English (Return only the translated text block without any introductory phrases and explanations!):"
            },
            {
                "role": "user",
                "content": candidate_info,
            }
        ],
        model="llama3-8b-8192",
    )

    # print(chat_completion.choices[0].message.content)

    return chat_completion.choices[0].message.content

def translate_text(text_to_translate):
    ''' 
    Translate the given text to English
    '''
    # print('translating text')
    # input_text = f'''
    #     Translate this text to English:
    #     {text_to_translate}

    #     Return only the translated text block without any introductory phrases and explanations!
    # '''

    chat_completion = GROQ_CLIENT.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Translate this text to English (Return only the translated text block without any introductory phrases and explanations!):"
            },
            {
                "role": "user",
                "content": text_to_translate,
            }
        ],
        model="llama3-8b-8192",
    )

    # print(chat_completion.choices[0].message.content)

    return chat_completion.choices[0].message.content

def process_candidates(candidates):
    '''
    Translate entries from other languages to English
    '''
    # print('start process candidates')

    for candidate in candidates:
        candidate_info = utils.exctract_info_for_summary(candidate['data'])
        candidate['data']['info'] = candidate_info

        try:
            curr_summary = candidate['data']['summary']
        except KeyError:
            curr_summary = generate_summary(candidate['data'])
            candidate['data']['summary'] = curr_summary
        try:
            if detect(curr_summary) != 'en':
                # translated_summary = pipe(curr_summary)
                translated_summary = translate_text(curr_summary)
                candidate['data']['summary'] = translated_summary
        except:
            print(curr_summary)    
        
    return candidates

def get_summary_embeddings(candidates, model_embedding):
    '''
    Get embeddings of candidate summaries that will be used as the keys in the vector database
    '''
    embeddings = model_embedding.embed_documents([x['data']['summary'] for x in candidates])
    return embeddings

def insert_record(collection, candidate, embedding):
    '''
    Insert record into ChromaDB
    '''
    metadata = {
        'candidate_summary': str(candidate['data']['summary']),
        'candidate_info': str(candidate['data']['info']),
        'candidate_number': str(candidate['file_number']),
        'candidate_full_name': str(candidate['data']['fullName'])
    }

    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())],
    )

    return None

def save_to_vector_db(candidates, embeddings, db_name=db_config.DB_NAME, collection_name=db_config.COLLECTION_NAME):
    '''
    Save candidates records to ChromaDB vector database
    '''
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), db_name))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for candidate, embedding in zip(candidates, embeddings):
        insert_record(collection, candidate, embedding)

    return None

def main(model_embedding):
    '''
    Main function
    '''
    if os.path.exists(os.path.join(os.getcwd(), 'candidates.pkl')):
        with open(os.path.join(os.getcwd(), 'candidates.pkl'), 'rb') as f:
            candidates = pickle.load(f)
    else:
        candidates = process_candidates(load_candidate_files(data_path))

    if os.path.exists(os.path.join(os.getcwd(), 'embeddings.pkl')):
        with open(os.path.join(os.getcwd(), 'embeddings.pkl'), 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_summary_embeddings(candidates, model_embedding)

    save_to_vector_db(candidates, embeddings)

    return None


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data')

    model_embedding = HuggingFaceEmbeddings(model_name=models_config.EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})

    main(model_embedding)


