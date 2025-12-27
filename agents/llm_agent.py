import os
import json
import urllib.request
import urllib.error
import re
from gym_env.enums import Action


class LLMAgent:
    """
    Agent that uses an LLM via OpenRouter to decide actions.
    The observation passed to action() is treated as the prompt.
    """

    def __init__(self, name="LLMAgent", model="deepseek/deepseek-r1"):
        self.name = name
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print(
                f"Warning: OPENROUTER_API_KEY environment variable not set for {self.name}."
            )

    def action(self, action_space, observation, info):
        """
        Decide action based on LLM response.
        observation: The prompt string for the LLM.
        """
        # The user specified: "observation now becomes llm_prompt"
        prompt = str(observation)

        # Get legal action values (integers)
        legal_actions = [a.value for a in action_space] if action_space else []

        if not self.api_key:
            # Fallback if no key
            return (
                0,
                {
                    "agent_type": "llm",
                    "error": "missing_api_key",
                    "legal": 1 if 0 in legal_actions else 0,
                },
            )

        try:
            response_text = self._call_llm(prompt)

            # Simple parsing: look for the first number
            match = re.search(r"\d+", response_text)
            if match:
                action_val = int(match.group())
            else:
                action_val = 99

            is_legal = 1 if action_val in legal_actions else 0

            return action_val, {
                "agent_type": "llm",
                "llm_response": response_text,
                "legal": is_legal,
            }

        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback on error
            return (
                0,
                {
                    "agent_type": "llm",
                    "error": str(e),
                    "legal": 1 if 0 in legal_actions else 0,
                },
            )

    def _call_llm(self, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PyPokerEngine/DeepHoldem",  # Optional, good practice for OpenRouter
            "X-Title": "DeepHoldem Agent",  # Optional
        }
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}

        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("Invalid response from OpenRouter: " + str(result))
