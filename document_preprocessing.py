import os
import fitz

from groq import Groq

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GROQ_CLIENT = Groq(
    api_key=GROQ_API_KEY,
)

def read_document(document_path):
    '''
    Read text frmo pdf document
    '''
    document = fitz.open(document_path)

    pdf_text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pdf_text += page.get_text()

    return pdf_text

def summarize_document(document_text):
    '''
    Summarize contents of document
    '''
    chat_completion = GROQ_CLIENT.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Summarize the info from the given document, select most important information and present it in 2-3 paragraphs. \
                            Return only the result text without any introductory phrases or additional information."
            },
            {
                "role": "user",
                "content": document_text,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

def summarize_document_into_profile(document_text):
    '''
    Summarize contents of document
    '''
    chat_completion = GROQ_CLIENT.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Summarize the info from the given document and generate a profile of candidate that will be most suitable for the event, described in the document. \n \
                            The profile summary should be something like this: \
                            With the skills and knowledge to adapt to the rapidly transforming technical and commercial environments, I am passionate about delivering leading technology innovations to the market and achieving successful commercial outcomes. I have extensive experience in risk management and am known for providing pragmatic commercial advice as well as a strategic approach to the protection, commercialisation and management of intellectual property assets. \n \
                            Return only the result text without any introductory phrases or additional information. \
                            "
            },
            {
                "role": "user",
                "content": document_text,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content


if __name__ == '__main__':
    document_text = read_document(os.path.join(os.getcwd(), 'test_document.pdf'))

    print(summarize_document(document_text))
    print('---')
    print(summarize_document_into_profile(document_text))