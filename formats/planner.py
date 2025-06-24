
from pydantic import BaseModel, Field
 

class plannerresponse(BaseModel):
    response: str = Field(..., description="The response from the planner agent, which includes the plan to be executed.")
    def format(self):
        return "\n".join([
            "**Instructions:**",
            "",
            self.response,
            "",
            "\n Please execute the plan as instructed. in depth",
        ])
    

