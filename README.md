# ag2_Cosmo

A multi-agent framework for cosmology research and simulation planning, built on AutoGen and OpenAI's API. Cosmogents orchestrates specialized AI agents to plan, review, and implement cosmological simulations through an iterative workflow with integrated RAG (Retrieval-Augmented Generation) capabilities.

## Architecture

### Core Agents

- **Planner**: Creates detailed plans for cosmology simulations
- **Reviewer**: Provides feedback and suggestions for plan improvements
- **Coder**: Implements the finalized plans in executable code
- **Admin**: Coordinates workflow and manages tool execution

### Knowledge Stores

Three specialized knowledge stores power the RAG functionality:

- **Coding**: Programming and implementation knowledge
- **Planck**: Planck satellite data and observations
- **Theory**: Theoretical cosmology concepts and models

## Quick Start

```python
from cosmogents import Cosmogents

# Initialize with your OpenAI API key
api_key = "your-openai-api-key"
planning_rounds = 3
cosmogents = Cosmogents(api_key=api_key, N_PLANNING_ROUNDS=planning_rounds)

# Set up vector stores (optional, for RAG functionality)
cosmogents.create_vector_store("path/to/your/data")

# Create agents
cosmogents.create_agent(temperature=0.3)

# Execute a cosmology simulation request
user_prompt = "Create a simulation to study dark matter halo formation"
result, context, last_agent = cosmogents.execute(user_prompt)
```

