"""
Prefix configuration for the prefix-aware inference gateway.

Maintains reusable reference prompts with exact token counts.
"""

from transformers import AutoTokenizer

from config.benchmark_config import SHARED_SYSTEM_PROMPT_BASE


# Model tokenizer used for prefix construction and verification
SHARED_PROMPT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PREFIX_TOKEN_COUNTS = {
    "shared_1500": 1500,
    "shared_2000": 2000,
}

_tokenizer = AutoTokenizer.from_pretrained(
    SHARED_PROMPT_MODEL,
    trust_remote_code=True,
)


def make_exact_shared_prefix(token_count: int) -> str:

    text = SHARED_SYSTEM_PROMPT_BASE.strip()

    token_ids = _tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    extensions = [
        " For every recommendation, identify its operational consequence.",
        " Track queue depth, active requests, and rejected requests separately.",
        " Preserve identical shared-prefix text across all benchmark requests.",
        " Report the measurement method and limitations alongside each result.",
        " Treat missing telemetry as unknown rather than interpreting it as zero.",
        " Compare workloads only when their relevant input conditions are equivalent.",
        " Keep request-specific identifiers and questions after the shared prefix.",
    ]

    index = 0

    while len(token_ids) < token_count:
        text += extensions[index % len(extensions)]
        index += 1

        token_ids = _tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    token_ids = token_ids[:token_count]

    while True:

        candidate = _tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        verified_ids = _tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )

        if len(verified_ids) == token_count:
            return candidate

        if len(verified_ids) > token_count:
            token_ids = token_ids[:token_count - 1]

        else:
            token_ids = verified_ids + _tokenizer.encode(
                " operational guidance",
                add_special_tokens=False,
            )

            token_ids = token_ids[:token_count]


# Generate reusable prefix profiles
PREFIX_PROFILES = {
    profile: make_exact_shared_prefix(token_count)
    for profile, token_count in PREFIX_TOKEN_COUNTS.items()
}


def get_prefix(profile: str) -> str | None:

    if profile == "none":
        return None

    if profile not in PREFIX_PROFILES:
        raise ValueError(
            f"Unknown prefix profile: {profile}"
        )

    return PREFIX_PROFILES[profile]


def verify_prefix_profiles():

    for profile, prefix in PREFIX_PROFILES.items():

        expected = PREFIX_TOKEN_COUNTS[profile]

        actual = len(
            _tokenizer.encode(
                prefix,
                add_special_tokens=False,
            )
        )

        assert actual == expected, (
            f"{profile}: expected {expected}, got {actual}"
        )

        print(
            f"{profile}: verified {actual} tokens"
        )