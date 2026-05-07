import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()

    api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
    endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
    model = os.getenv("AZURE_FOUNDRY_MODEL", "").strip('"')

    missing = [
        name
        for name, value in {
            "AZURE_FOUNDRY_API_KEY": api_key,
            "AZURE_FOUNDRY_ENDPOINT": endpoint,
            "AZURE_FOUNDRY_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = OpenAI(api_key=api_key, base_url=endpoint)

    prompt = "How many R's are there in the word raspberry? Return just the number."
    response = client.responses.create(model=model, input=prompt)

    print("Prompt:", prompt)
    print("Model:", model)
    print("Answer:", response.output_text)


if __name__ == "__main__":
    main()
