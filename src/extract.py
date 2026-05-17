import json
import logging

logger = logging.getLogger(__name__)

EXTRACTION_TEMPLATE = """From the {filing_type} filing text below, extract the value for: "{variable}"

Rules:
- For numeric values, return the plain number only — no units, no currency symbols (e.g. return "3,701" not "$3,701 million")
- For numeric values, prefer the most recent fiscal year's consolidated figure
- For segment names, list the operating sub-segments (not top-level groups) separated by commas (e.g. "North America, International, Individual Retirement")
- If the value is not found, return null

Respond ONLY with a JSON object in this exact format:
{{"variable": "{variable}", "value": "<extracted value or null>", "context": "<short quote from text showing where you found it, or null>"}}

Text:
{context}
"""


def _system_prompt(filing_type):
    return (
        f"You are a financial document analyst. Your job is to extract specific "
        f"financial values from SEC {filing_type} filing excerpts. Be precise. "
        f"Return only what is explicitly stated in the text."
    )


def extract_variable(variable, retrieved_chunks, client, model="gpt-4o-mini", filing_type="10-K"):
    context_text = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = EXTRACTION_TEMPLATE.format(
        filing_type=filing_type, variable=variable, context=context_text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt(filing_type)},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM extraction failed for '{variable}': {e}")
        result = {"variable": variable, "value": None, "context": None}

    logger.info(f"Extracted '{variable}' → {result.get('value')}")
    return result


def extract_all_variables(variables, chunks_per_variable, client, model="gpt-4o-mini", filing_type="10-K"):
    results = []
    for var in variables:
        chunks = chunks_per_variable.get(var, [])
        if not chunks:
            logger.warning(f"No chunks for '{var}', skipping")
            results.append({"variable": var, "value": None, "context": None})
            continue
        results.append(extract_variable(var, chunks, client, model, filing_type))
    return results
