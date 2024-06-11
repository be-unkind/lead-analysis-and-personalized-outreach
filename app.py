import gradio as gr

import os
import utils
import torch
from document_preprocessing import read_document, summarize_document, summarize_document_into_profile
import multiagent_system
from configs import db_config, models_config

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_EMBEDDING = HuggingFaceEmbeddings(model_name=models_config.EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})

def main(document):
    '''
    Main pipeline
    '''
    document_text = read_document(document)
    
    document_summary = summarize_document(document_text)
    perfect_profile = summarize_document_into_profile(document_text)

    top_candidates = utils.find_top_candidates(MODEL_EMBEDDING, perfect_profile, n_results=5)

    agents = multiagent_system.initialize_agents()
    result = multiagent_system.personalized_emails(agents, document_summary, top_candidates)

    print(result)
    return result


def define_gradio_ui():
    iface = gr.Interface(
        fn=main,
        inputs=[gr.File()],
        outputs="text",
        title="System for lead analysis and personalized outreach",
        description="Receive suggestions about personalized emails for top candidates"
        )

    iface.launch(share=True)

if __name__ == "__main__":
    define_gradio_ui()