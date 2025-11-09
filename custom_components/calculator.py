from simpleeval import simple_eval
from langflow.custom.custom_component.component import Component
from langflow.io import Output, MessageTextInput
from math import sqrt, log, sin, cos, tan, exp


class CalculatorComponent(Component):
    MATHS_FUNCTIONS = {
        'sqrt': sqrt,
        'log': log,
        'sin': sin,
        'cos': cos,
        'tan': tan,
        'exp': exp
    }
    display_name = "Calculator"
    description = "A component that evaluates mathematical expressions."
    icon = "calculator"
    name = "CalculatorComponent"

    inputs=[
        MessageTextInput(
            name="expression",
            display_name="Mathematical Expression",
            info="The mathematical expression to evaluate.",
            tool_mode=True,
        )
    ]
    outputs=[
        Output(
            name="result",
            display_name="Result",
            method="calculate",
        )
    ]

    def calculate(self):
        try:
            result = simple_eval(self.expression, functions=self.MATHS_FUNCTIONS)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"