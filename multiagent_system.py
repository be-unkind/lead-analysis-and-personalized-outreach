import autogen
import os

from configs import db_config, models_config

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

config_list = [
    {
        "model": models_config.MODEL_NAME_GROQ,
        "base_url": models_config.BASE_URL_GROQ,
        "api_key": GROQ_API_KEY,
    }
]

LLAMA_CONFIG = {"config_list": config_list}


def initialize_agents():
    '''
    Initialize agent for the sequential chat
    '''
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="A proxy for the user for initiating chat with provided question.",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    argument_choices = autogen.ConversableAgent(
        name="Agent 1: Argument choice of each of the candidates",
        system_message=""" 
            Provide argumentation (5-6 sentences) on why each of the candidates should be interested or attend the event
            In this argumentation, include a few words about what is this event, what is it about, etc.

            Keep structure like this in the response:

            - Candidate full name, id and info
            - Argumentation

            - Candidate full name, id and info
            - Argumentation

            and so on ...

            Return only the result, without any introductory phrases or additional text.
        """,
        llm_config=LLAMA_CONFIG,
        human_input_mode="NEVER",
    )

    generate_personalized_messages = autogen.ConversableAgent(
        name="Agent 2: Write personalized messages",
        system_message="""
            Analyze all of the candidate infos and argumentations on why they shoul attend the event or be interested in it and provide personalized message for each of them based on a template
            In the body of the personalized message you should include the argumentation on why the person should attend or be interested in the event

            Personalized message template:

            Hello John Doe,
            
            Tell a little bit about the event
            Generated argumentation on why the person should attend

            Thanks for your attention, will be glad to see you with us!
            Have a great day,
            Team

            you should return as a result the personalized messages for all of the candidates along with their info

            Returned result should look like this:
            
            Candidate full name, id and info
            Personalized message

            ---

            Candidate full name, id and info
            Personalized message

            ---

            and so on...

            Return only the result, without any introductory phrases or additional text.
        """,
        llm_config=LLAMA_CONFIG,
        human_input_mode="NEVER" 
    )

    return [user_proxy, argument_choices, generate_personalized_messages]

def personalized_emails(agents, document_summary, top_candidates):
    '''
    Initialize sequential chat to generate personalized messages
    '''
    top_candidates_str = ""
    for top_c in top_candidates:
        top_candidates_str += f"""
            Id: {top_c['candidate_number']}
            Full name: {top_c['candidate_full_name']},
            Info: {top_c['candidate_info']}
            Summary: {top_c['candidate_summary']}
            ---
        """

    chat_results = agents[0].initiate_chats(
        [
            {
                "recipient": agents[1],
                "message": f"""
                    This is the summary of an event:
                    {document_summary}

                    These are the info on top candidates that may be interested in visiting this event/course:
                    {top_candidates_str}
                    ---
                """,
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": agents[2],
                "message": "These are the descriptions of the candidates along with the argumentation on why they should attent the event/course or why they may be interested in it",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
        ]
    )

    return chat_results[-1].chat_history[-1]['content']    
