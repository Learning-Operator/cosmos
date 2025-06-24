from pydantic import BaseModel, Field


class reviewerresponse(BaseModel):
    response: str = Field(..., description="The response from the reviewer agent, which includes the plan to be executed.")
    def format(self):
        return "\n".join([
            "**Instructions:**",
            "",
            self.response,
            "",
            "\n Please execute the plan as instructed. in depth",
        ])