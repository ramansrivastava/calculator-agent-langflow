# file: agent_runner_component.py

from typing import List
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output, IntInput, MultilineInput, HandleInput, BoolInput
from langflow.schema import Data, Message
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langflow.logging import logger

class AgentRunnerComponent(Component):
    display_name = "Agent Runner (N-times)"
    description = "Runs an agent N times on the same question using the provided LLM and tools, then returns all responses for majority voting."
    icon = "fa-solid fa-redo"
    name = "AgentRunnerComponent"

    inputs = [
        HandleInput(
            name="question",
            display_name="Question",
            info="The question to ask the Agent",
            input_types=["Message"],
        ),
        IntInput(
            name="n_runs",
            display_name="Number of Runs",
            info="How many times to run the Agent for generating responses",
            value=3,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            info="The language model to use for the agent",
            input_types=["LanguageModel"],
        ),
        HandleInput(
            name="tools",
            display_name="Tools",
            info="List of tools the agent can use (e.g., Calculator)",
            input_types=["Tool"],
            is_list=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="System Prompt",
            info="Instructions for the agent",
            value="You are a helpful assistant that can use tools to answer questions accurately.",
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Mode",
            info="Enable detailed logging for debugging",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="responses",
            display_name="Responses List",
            method="run_agent_multiple",
        )
    ]

    def _extract_question_text(self) -> str:
        """Extract text from Message object or other input types."""
        if isinstance(self.question, Message):
            return self.question.text
        elif hasattr(self.question, "content"):
            return self.question.content
        return str(self.question)

    async def run_agent_multiple(self) -> Data:
        """Run the agent N times and collect all responses."""
        question_text = self._extract_question_text()
        n = max(1, self.n_runs)

        if self.verbose:
            await logger.ainfo(f"Running agent {n} times for: {question_text}")

        all_responses: List[str] = []
        for i in range(n):
            try:
                answer = await self.invoke_agent(question_text)
                all_responses.append(answer)
                if self.verbose:
                    await logger.ainfo(f"Run {i+1}/{n}: {answer}")
            except Exception as e:
                error_msg = f"Error in run {i+1}: {str(e)}"
                await logger.aerror(error_msg)
                all_responses.append(error_msg)

        return Data(data={"responses": all_responses})

    async def invoke_agent(self, question: str) -> str:
        """Invoke agent once and return the answer."""
        try:
            # Create agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=15,
                return_intermediate_steps=True,
            )

            # Invoke agent
            result = await agent_executor.ainvoke({"input": question})

            # Extract output
            output = None
            if isinstance(result, dict):
                output = result.get("output", "").strip() or None

                # Fallback to intermediate steps if output is empty
                if not output and "intermediate_steps" in result:
                    steps = result["intermediate_steps"]
                    if steps and len(steps) > 0 and isinstance(steps[-1], tuple) and len(steps[-1]) >= 2:
                        output = str(steps[-1][1]).strip()
            elif hasattr(result, "content"):
                output = result.content.strip()
            else:
                output = str(result).strip()

            return output if output else "No answer generated"

        except Exception as e:
            await logger.aerror(f"Error in invoke_agent: {str(e)}", exc_info=True)
            raise

