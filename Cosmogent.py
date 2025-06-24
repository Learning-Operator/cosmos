from openai import OpenAI
from pathlib import Path
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter
from autogen import LLMConfig
import copy
import os
from pydantic import BaseModel, Field
from autogen import ConversableAgent, UpdateSystemMessage, UserProxyAgent
from autogen.agentchat import a_initiate_group_chat, initiate_group_chat, register_function
from autogen.agentchat.group import (
    AgentTarget,
    ContextVariables,
    ReplyResult,
    OnCondition,
    StringLLMCondition,
    TerminateTarget,
)
from autogen.agentchat.group.patterns import (
    DefaultPattern,
)
import os
import shutil
from typing import Annotated, List, Tuple, Literal
from autogen import ConversableAgent, LLMConfig
import os
from IPython.display import display, Markdown
import yaml
import time

from formats.planner import plannerresponse
from formats.reviewer import reviewerresponse
from formats.coder import Coderresponse

from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent


class Cosmogents:    
    MAX_CONTEXT = 6
    
    def __init__(self, api_key: str, N_PLANNING_ROUNDS: int, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.N_ROUNDS = N_PLANNING_ROUNDS
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.stores = []
        self.agents = []
        self.assistant_ids = []
        self.rag_agents = []

    def execute(self, user_prompt):
        if not self.agents:
            raise ValueError("Agents not created. Call create_agent() first.")
            
        planner = self.agents[0]
        reviewer = self.agents[1] 
        coder = self.agents[2]
        admin = self.agents[3]
        
        workflow_context = ContextVariables(
            data={
                "Planning_round": 0,
                "planner_history": [],
                "reviewer_history": [],
            }
        )
        
        agent_pattern = DefaultPattern(
            agents=self.agents,
            initial_agent=planner,
            context_variables=workflow_context,
        )
        
        def record_steps(agent_choice: str, context_variables: ContextVariables) -> ReplyResult:
            context_variables["Planning_round"] += 0.5
            half_round = context_variables["Planning_round"]

            if half_round % 1 == 0.5:
                context_variables["planner_history"].append(agent_choice)
                display(Markdown(agent_choice))

                if half_round >= self.N_ROUNDS:
                    return pass_to_coder(context_variables)

                return ReplyResult(
                    target=AgentTarget(reviewer),
                    message="Please review the plan and provide feedback.",
                    context_variables=context_variables,
                )
            
            else:
                context_variables["reviewer_history"].append(agent_choice)
                display(Markdown(f"{agent_choice}"))

                if half_round >= self.N_ROUNDS:
                    self.N_ROUNDS += 0.5

                return ReplyResult(
                    target=AgentTarget(planner),
                    message="Please update the plan based on the review.",
                    context_variables=context_variables,
                )
            
        def pass_to_coder(context_variables: ContextVariables) -> ReplyResult:
            planner_history = context_variables.get("planner_history", [])
            if planner_history:
                final_plan = planner_history[-1]
            else:
                final_plan = "No plan available"

            return ReplyResult(
                target=AgentTarget(coder),  
                message=final_plan,
                context_variables=context_variables,
            )
        
        def prompt_rag(query: str, store_type: Literal['Coding', 'planck', 'theory']) -> str:
            try:
                store_map = {"Coding": 0, "planck": 1, "theory": 2}
                if store_type not in store_map:
                    return f"Invalid store type: {store_type}. Available types: Coding, planck, theory"
                
                store_index = store_map[store_type]
                if store_index >= len(self.assistant_ids):
                    return f"RAG assistant for {store_type} not available"
                
                assistant_id = self.assistant_ids[store_index]
                
                thread = self.client.beta.threads.create()
                
                self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=f"Search for information about: {query}"
                )
                
                run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=assistant_id
                )
                
                if run.status == 'completed':
                    messages = self.client.beta.threads.messages.list(
                        thread_id=thread.id,
                        order="desc",
                        limit=1
                    )
                    
                    if messages.data:
                        content = messages.data[0].content[0].text.value
                        return content
                    else:
                        return "No response from assistant"
                else:
                    return f"Assistant run failed with status: {run.status}"
                    
            except Exception as e:
                return f"Error querying RAG system: {str(e)}"

        register_function(
            record_steps,
            caller=admin,
            executor=admin,
            name="record_step",
            description="Track planner and reviewer messages over multiple rounds, manage turn-taking, and pass the final plan to the coder when rounds complete.",
        )

        register_function(
            pass_to_coder,
            caller=admin,
            executor=admin,
            name="pass_to_coder",
            description="Send the finalized plan, compiled from planner and reviewer outputs, directly to the coder agent for implementation.",
        )

        register_function(
            prompt_rag,
            caller=admin,
            executor=admin,
            name="prompt_rag",
            description="Search the cosmology knowledge base for relevant research information, theories, and data. Use store_type: 'Coding' for programming info, 'planck' for Planck satellite data, 'theory' for theoretical cosmology.",
        )

        initial_messages = [{"role": "user", "content": user_prompt}]
            
        chat_result, context_variables, last_agent = initiate_group_chat(
            pattern=agent_pattern,
            messages=initial_messages,
            max_rounds=500,
        )
        return chat_result, context_variables, last_agent

    def create_rag_agents(self):
        if not self.stores:
            print("Warning: No vector stores available for RAG agents")
            return
            
        self.assistant_ids = []

        def create_oai_assistant(vector_store_id: str, store_name: str) -> str:
            try:
                rag_agent = self.client.beta.assistants.create(
                    name=f"Rag_agent_{store_name}",
                    instructions=f"""You are a helpful cosmology research assistant specialized in {store_name} knowledge. 
                             When answering questions, search through the available documents to provide accurate, detailed responses 
                             based on the uploaded research materials. Always cite relevant information from the documents.""",
                    tools=[{"type": "file_search", "file_search": {"max_num_results": 5}}],
                    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
                    model=self.model,
                    temperature=0.0,
                    top_p=1.0,
                    response_format="auto"
                )
                return rag_agent.id
            except Exception as e:
                print(f"Error creating assistant for {store_name}: {e}")
                return None

        store_names = ["Coding", "planck", "theory"]
        for i, store in enumerate(self.stores):
            store_name = store_names[i] if i < len(store_names) else f"store_{i}"
            assistant_id = create_oai_assistant(store.id, store_name)
            if assistant_id:
                self.assistant_ids.append(assistant_id)

        print(f"Created {len(self.assistant_ids)} RAG assistants")

    def create_agent(self, temperature: float = 0.3):
        
        self.llm_config_oai = LLMConfig(
            api_type="openai",
            model=self.model,
            api_key=self.api_key,
            temperature=temperature,
        )
        
        planner_config = copy.deepcopy(self.llm_config_oai)
        planner_config.response_format = plannerresponse

        reviewer_config = copy.deepcopy(self.llm_config_oai)
        reviewer_config.response_format = reviewerresponse

        coder_config = copy.deepcopy(self.llm_config_oai)
        coder_config.response_format = Coderresponse

        try:
            planner_system_message = yaml.safe_load(Path("formats/planner.yaml").read_text())['instructions']
            reviewer_system_message = yaml.safe_load(Path("formats/reviewer.yaml").read_text())['instructions']
            coder_system_message = yaml.safe_load(Path("formats/coder.yaml").read_text())['instructions']
        except FileNotFoundError as e:
            print(f"Error loading system message files: {e}")
            planner_system_message = "You are a planning agent that creates detailed plans for cosmology simulations."
            reviewer_system_message = "You are a review agent that provides feedback on plans."
            coder_system_message = "You are a coding agent that implements plans in code."

        planner_history_template = "\n\nPrevious planning history:\n{planner_history}"
        reviewer_history_template = "\n\nPrevious reviewer history:\n{reviewer_history}"

        admin_llm_config = copy.deepcopy(self.llm_config_oai)
        admin_llm_config.parallel_tool_calls = False

        game_master_system_message = """
        You are the Admin. You coordinate the workflow between agents and manage tool calls.
        You keep track of the number of rounds and facilitate communication between agents.
        When agents need to use tools, you execute them and return the results.
        You should help other agents by calling the appropriate functions when they need information.
        """.strip()

        admin = ConversableAgent(
            name="Admin",
            system_message=game_master_system_message,
            llm_config=admin_llm_config,
            human_input_mode="NEVER",
        )

        planner = ConversableAgent(
            name="Planner",
            llm_config=planner_config,
            human_input_mode="NEVER",
            update_agent_state_before_reply=[
                UpdateSystemMessage(planner_system_message + planner_history_template),
            ],
        )

        reviewer = ConversableAgent(
            name="Reviewer",
            llm_config=reviewer_config,
            human_input_mode="NEVER",
            update_agent_state_before_reply=[
                UpdateSystemMessage(reviewer_system_message + reviewer_history_template),
            ],
        )

        coder = ConversableAgent(
            name="Coder",
            llm_config=coder_config,
            human_input_mode="NEVER",
            update_agent_state_before_reply=[
                UpdateSystemMessage(coder_system_message),
            ],
        )

        context_handling_planner = TransformMessages(
            transforms=[
                MessageHistoryLimiter(
                    max_messages=self.MAX_CONTEXT, 
                    keep_first_message=True,
                    exclude_names=["Admin", "Reviewer", "_Group_Tool_Executor"],
                ),
            ]
        )
        context_handling_planner.add_to_agent(planner)

        context_handling_reviewer = TransformMessages(
            transforms=[
                MessageHistoryLimiter(
                    max_messages=self.MAX_CONTEXT, 
                    keep_first_message=True,
                    exclude_names=["Admin", "Planner", "_Group_Tool_Executor"],
                ),
            ]
        )
        context_handling_reviewer.add_to_agent(reviewer)

        context_handling_coder = TransformMessages(
            transforms=[
                MessageHistoryLimiter(
                    max_messages=self.MAX_CONTEXT, 
                    keep_first_message=True,
                    exclude_names=["Admin", "_Group_Tool_Executor"],
                ),
            ]
        )
        context_handling_coder.add_to_agent(coder)

        planner.handoffs.set_after_work(AgentTarget(admin))
        reviewer.handoffs.set_after_work(AgentTarget(admin))
        coder.handoffs.set_after_work(AgentTarget(admin))

        self.agents = [planner, reviewer, coder, admin]

        self.create_rag_agents()

        for agent in self.agents:
            agent.reset()

        print(f"Created {len(self.agents)} main agents and {len(self.assistant_ids)} RAG assistants")

    def create_vector_store(self, directory_path: str = "data"):
        
        chunking_strategy = {
            "type": 'static',
            'static': {
                "max_chunk_size_tokens": 400,
                "chunk_overlap_tokens": 200
            }
        }
        
        start = time.time()
        assistant_data = self.load_directory()

        print("Creating vector stores...")
        
        for typestore in ["Coding", "planck", "theory"]:
            store_path = os.path.join(assistant_data, typestore)
            
            if not os.path.exists(store_path):
                print(f"Warning: Directory {store_path} does not exist. Skipping {typestore} store.")
                continue
                
            print(f"\nProcessing {typestore} store...")
            print("Files to upload:")
            
            file_paths = []
            for root, dirs, files in os.walk(store_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if file.startswith('.'):
                        continue
                    file_path = os.path.join(root, file)
                    print(f" - {file}")
                    file_paths.append(file_path)

            if not file_paths:
                print(f"No files found in {store_path}. Creating empty store.")
                try:
                    vector_store = self.client.vector_stores.create(
                        name=f"{typestore}_store",
                        chunking_strategy=chunking_strategy
                    )
                    self.stores.append(vector_store)
                except Exception as e:
                    print(f"Error creating empty vector store for {typestore}: {e}")
                continue

            try:
                vector_store = self.client.vector_stores.create(
                    name=f"{typestore}_store",
                    chunking_strategy=chunking_strategy
                )

                streams = [open(path, "rb") for path in file_paths]
                try:
                    file_batch = self.client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id,
                        files=streams
                    )
                    print(f"Successfully uploaded {len(file_paths)} files to {typestore} store")
                    print(f"File batch status: {file_batch.status}")
                finally:
                    for s in streams:
                        s.close()

                self.stores.append(vector_store)
                
            except Exception as e:
                print(f"Error creating vector store for {typestore}: {e}")
                continue

        end = time.time()
        print(f"\nVector store creation completed in {end - start:.2f} seconds")
        print(f"Created {len(self.stores)} vector stores")

    def load_directory(self) -> str:
        os.makedirs("data", exist_ok=True)
        return os.path.abspath("data")

