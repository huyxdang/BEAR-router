import time

import anthropic
import openai
from google import genai
from google.genai import types

from config import ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise and give the answer directly."
)

# Sync clients
_anthropic_client = None
_openai_client = None
_google_client = None

# Async clients
_async_anthropic_client = None
_async_openai_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_async_anthropic_client():
    global _async_anthropic_client
    if _async_anthropic_client is None:
        _async_anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _async_anthropic_client


def _get_google_client():
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _google_client


def _get_async_openai_client():
    global _async_openai_client
    if _async_openai_client is None:
        _async_openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _async_openai_client


def call_llm(model_config: dict, prompt_text: str) -> dict:
    """Call an LLM and return response with metadata.

    Returns dict with: response_text, input_tokens, output_tokens, latency.
    """
    provider = model_config["provider"]
    model_id = model_config["id"]

    start = time.time()

    if provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=256,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency = time.time() - start
        text = response.content[0].text if response.content else ""
        return {
            "response_text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }

    elif provider == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=256,
            temperature=0,
        )
        latency = time.time() - start
        return {
            "response_text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency,
        }

    elif provider == "google":
        client = _get_google_client()
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=256,
            ),
        )
        latency = time.time() - start
        return {
            "response_text": response.text or "",
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "latency": latency,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def call_llm_async(model_config: dict, prompt_text: str) -> dict:
    """Async version of call_llm."""
    provider = model_config["provider"]
    model_id = model_config["id"]

    start = time.time()

    if provider == "anthropic":
        client = _get_async_anthropic_client()
        response = await client.messages.create(
            model=model_id,
            max_tokens=256,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency = time.time() - start
        return {
            "response_text": response.content[0].text if response.content else "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }

    elif provider == "openai":
        client = _get_async_openai_client()
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=256,
            temperature=0,
        )
        latency = time.time() - start
        return {
            "response_text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency,
        }

    elif provider == "google":
        client = _get_google_client()
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=256,
            ),
        )
        latency = time.time() - start
        return {
            "response_text": response.text or "",
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "latency": latency,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")
