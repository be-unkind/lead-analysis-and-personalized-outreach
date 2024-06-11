import os
import chromadb
from configs import db_config, models_config

def exctract_info_for_summary(data):
    '''
    Extract info from candidate file
    '''
    info = ''

    industry = data.get('industry')
    if industry:
        info += f'Industry: {industry}\n'

    current_positions = data.get('currentPositions')
    if current_positions:
        current_position = current_positions[0]
        title = current_position.get('title')
        company_name = current_position.get('companyName')
        if title and company_name:
            info += f'Current work position: {title} at the {company_name}'
            company_industry = current_position.get('companyUrnResolutionResult', {}).get('industry')
            if company_industry:
                info += f' in the industry of {company_industry}'
            info += '.\n'

    educations = data.get('educations')
    if educations:
        # educations = educations[1:-1] if len(educations) > 1 else educations
        for edu in educations:
            degree = edu.get('degree')
            fields_of_study = edu.get('fieldsOfStudy')
            if degree and fields_of_study:
                info += f'Education: {degree} in the field of {fields_of_study[0]}.\n'

    skills = data.get('skills')
    if skills:
        info += 'Skills: ' + ', '.join(skill['name'] for skill in skills) + '.\n'

    certifications = data.get('certifications')
    if certifications:
        info += 'Courses and certifications: '
        for cert in certifications:
            cert_name = cert.get('name')
            authority = cert.get('authority')
            if cert_name:
                info += '"' + cert_name + '"'
                if authority:
                    info += f' by {authority}'
                info += ', '
        info = info[:-2] + '.\n'
    
    return info

def find_top_candidates(model, query: str, n_results=db_config.DEFAULT_N_RESULTS, db_name=db_config.DB_NAME, collection_name=db_config.COLLECTION_NAME):
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), db_name))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    query_vector = model.embed_query(query)
    query_result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
    )

    top_candidates = []

    for res in query_result['metadatas'][0]:
        try:
            top_candidates.append(res)
        except:
            continue

    return top_candidates