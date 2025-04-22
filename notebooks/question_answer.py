import pydantic

class QuestionAnswer(pydantic.BaseModel):
    """Represents question - answer interaction."""
    question: str
    answer: str
