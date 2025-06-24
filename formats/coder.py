from pydantic import BaseModel, Field


class Coderresponse(BaseModel):
    response: str = Field(..., description="The response from the coder agent")
    explanation: str = Field(..., description="The explanation of the code provided by the coder agent")
    def format(self):
        return "\n".join([
            "**Code:**",
            "\n ******************************************************************************************",
            "",
            self.response,
            "",
            "\n ******************************************************************************************",
            "\n",
            self.explanation,
            "",
        ])