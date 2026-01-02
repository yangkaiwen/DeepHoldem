import os
import json
import time
import urllib.request
import urllib.error
import re
from gym_env.enums import Action


class LLMAgent:
    """
    Agent that uses an LLM via OpenRouter to decide actions.
    The observation passed to action() is treated as the prompt.
    """

    def __init__(self, name="LLMAgent", model="openai/gpt-4.1-mini"):
        self.name = name
        self.model = model
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

        if (
            not self.openrouter_api_key
            and not self.openai_api_key
            and not self.deepseek_api_key
        ):
            print(
                f"Warning: No API keys (OPENROUTER, OPENAI, DEEPSEEK) found for {self.name}."
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

        # Check if we have a valid key for the requested model
        use_openai_direct = self.model.startswith("openai/") and self.openai_api_key
        use_deepseek_direct = (
            self.model.startswith("deepseek/") and self.deepseek_api_key
        )
        has_key = use_openai_direct or use_deepseek_direct or self.openrouter_api_key

        if not has_key:
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
            try:
                response_text = self._call_llm(prompt)
            except Exception as e:
                print(
                    f"  [LLM Error] Primary model {self.model} failed: {e}. Fallback to deepseek/deepseek-chat"
                )
                response_text = self._call_llm(prompt, model="deepseek/deepseek-chat")

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

            # Smart Fallback: Try to CHECK (1) if legal, otherwise FOLD (0)
            fallback_action = 0  # Default FOLD
            fallback_name = "FOLD"

            if 1 in legal_actions:
                fallback_action = 1  # CHECK
                fallback_name = "CHECK"

            print(f"  -> Fallback: {fallback_name} ({fallback_action})")

            return (
                fallback_action,
                {
                    "agent_type": "llm",
                    "error": str(e),
                    "legal": 1,  # Fallback is always chosen from legal
                },
            )

    def _call_llm(self, prompt, model=None):
        target_model = model if model else self.model

        # Determine if we should use OpenAI direct
        use_openai_direct = target_model.startswith("openai/") and self.openai_api_key
        use_deepseek_direct = (
            target_model.startswith("deepseek/") and self.deepseek_api_key
        )

        if use_openai_direct:
            url = "https://api.openai.com/v1/chat/completions"
            api_key = self.openai_api_key
            # Strip 'openai/' prefix for direct API usage
            model_id = target_model.replace("openai/", "")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        elif use_deepseek_direct:
            url = "https://api.deepseek.com/chat/completions"
            api_key = self.deepseek_api_key
            # Strip 'deepseek/' prefix for direct API usage
            model_id = target_model.replace("deepseek/", "")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            api_key = self.openrouter_api_key
            model_id = target_model
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "DeepHoldem Agent",  # Optional
            }

        data = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a poker playing agent. You must respond with ONLY the index of the action you want to take. Do not provide any reasoning or explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 3,
        }

        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"), headers=headers
        )

        start_time = time.time()
        # Set a timeout for the request (e.g., 30 seconds)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

                elapsed = time.time() - start_time
                if elapsed > 2.0:
                    print(
                        f"  [Slow LLM Response] Model: {self.model}, Time: {elapsed:.2f}s"
                    )

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Invalid response from OpenRouter: " + str(result))
        except urllib.error.URLError as e:
            print(f"  [LLM Network Error] {e} (Time: {time.time() - start_time:.2f}s)")
            raise e
