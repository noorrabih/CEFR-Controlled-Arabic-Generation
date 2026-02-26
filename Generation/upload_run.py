import time
import json
from openai import OpenAI


client = OpenAI(api_key="key")  # expects OPENAI_API_KEY in env

BATCH_INPUT_PATH = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_prompt/6levels/batch_input.jsonl"
OUT_PATH = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_prompt/6levels/generated_essays_6levels.tsv"

# 1) Upload JSONL file
batch_file = client.files.create(
    file=open(BATCH_INPUT_PATH, "rb"),
    purpose="batch",
)

print("Uploaded file id:", batch_file.id)

# 2) Create batch job
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",  # only option for now
)

print("Batch id:", batch.id, "status:", batch.status)



# batch = client.batches.retrieve("batch_693bf16d672081908f056e7861ba29b6")

BATCH_ID = batch.id

# 1) Poll status (you can also just check in the dashboard)
while True:
    batch = client.batches.retrieve(BATCH_ID)
    print("Status:", batch.status)
    if batch.status in ("completed", "failed", "cancelled", "expired"):
        break
    time.sleep(10)

if batch.status != "completed":
    raise RuntimeError(f"Batch finished with status {batch.status}")

# 2) Get output file content
output_file_id = batch.output_file_id
result_stream = client.files.content(output_file_id)
result_text = result_stream.read().decode("utf-8")

# 3) Parse each result line and save essays as TSV
with open(OUT_PATH, "w", encoding="utf-8") as fout:
    # Write TSV header
    fout.write("Document_ID\tGrade\tessay\n")

    for line in result_text.splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)

        custom_id = obj["custom_id"]
        error = obj.get("error")
        if error:
            print("[ERROR in]", custom_id, error)
            continue

        response = obj["response"]["body"]
        essay = response["choices"][0]["message"]["content"]

        # custom_id = f"{prompt_id}_{level}_{i+1}"
        parts = custom_id.split("_")
        prompt_id = parts[0]
        level = parts[1]

        # Clean essay (important for TSV integrity)
        essay = essay.replace("\t", " ").replace("\n", " ").strip()

        fout.write(f"{custom_id}\t{level}\t{essay}\n")

print(f"✅ Wrote parsed essays to {OUT_PATH}")
