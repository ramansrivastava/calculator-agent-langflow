from collections import Counter
import re
from langflow.custom.custom_component.component import Component
from langflow.io import Output, DataInput
from langflow.schema import Message

class MajorityVotingComponent(Component):
    display_name = "Majority Voting"
    description = "Takes a list of responses and returns the majority answer."
    icon = "fa-solid fa-vote-yea"
    name = "MajorityVotingComponent"

    inputs = [
        DataInput(
            name="responses",
            display_name="List of Responses",
            is_list=True,
            input_types=["Data"],
        ),
    ]

    outputs = [
        Output(
            name="majority_answer",
            display_name="Majority Answer",
            method="compute_majority",
        )
    ]

    def _extract_number(self, response: str) -> str:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', str(response))
        return numbers[-1] if numbers else str(response).strip()

    def compute_majority(self) -> Message:
        responses = self.responses

        if isinstance(responses, list) and len(responses) == 1:
            responses = responses[0]

        if hasattr(responses, 'data') and isinstance(responses.data, dict):
            responses = responses.data.get('responses', [])

        if not isinstance(responses, list):
            responses = [responses]

        answers = [self._extract_number(r) for r in responses]

        if not answers:
            return Message(text="Error: no responses provided")

        counter = Counter(a.lower() for a in answers)
        winner, count = counter.most_common(1)[0]

        original = next((a for a in answers if a.lower() == winner), answers[0])
        confidence = (100 * count) // len(answers)

        responses_list = ", ".join(answers)
        return Message(text=f"{original} (confidence: {count}/{len(answers)} = {confidence}%)\nResponses: [{responses_list}]")
