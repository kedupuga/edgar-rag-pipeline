import json
import logging

logger = logging.getLogger(__name__)

EXTRACTION_TEMPLATE = """From the {filing_type} filing text below, extract ALL distinct values for: "{variable}"

Rules:
- Return every distinct value you find — do not pick just one
- For numeric values, return plain numbers only, no units or currency symbols (e.g. "49,746" not "$49,746 million")
- For segment names, return each segment name as a separate entry in the list
- Do not repeat the same value twice
- If no values are found, return an empty list

Respond ONLY with a JSON object in this exact format:
{{"variable": "{variable}", "values": ["value1", "value2", "..."], "context": "<short quote showing where you found the values>"}}

Text:
{context}
"""


def _system_prompt(filing_type):
    return (
        f"You are a financial document analyst. Extract specific financial values "
        f"from SEC {filing_type} filing excerpts. Return only what is explicitly "
        f"stated in the text. Be thorough — find all distinct occurrences."
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
        # handle models that return "value" instead of "values"
        if "values" not in result:
            v = result.get("value")
            result["values"] = [v] if v and str(v).lower() not in ("null", "none") else []
    except Exception as e:
        logger.error(f"LLM extraction failed for '{variable}': {e}")
        result = {"variable": variable, "values": [], "context": None}

    logger.info(f"Extracted '{variable}' -> {result.get('values')}")
    return result


def extract_all_variables(variables, chunks_per_variable, client, model="gpt-4o-mini", filing_type="10-K"):
    results = []
    for var in variables:
        chunks = chunks_per_variable.get(var, [])
        if not chunks:
            logger.warning(f"No chunks for '{var}', skipping")
            results.append({"variable": var, "values": [], "context": None})
            continue
        results.append(extract_variable(var, chunks, client, model, filing_type))
    return results
