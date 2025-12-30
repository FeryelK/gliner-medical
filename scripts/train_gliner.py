from gliner import GLiNER
import json

# ======================
# Load data
# ======================
print("📄 Loading data...")

with open("train_gliner_ready.json", encoding="utf-8") as f:
    train_data = json.load(f)

with open("valid_gliner_ready.json", encoding="utf-8") as f:
    valid_data = json.load(f)

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")

# ======================
# Load model
# ======================
print("🧠 Loading GLiNER model...")
model = GLiNER.from_pretrained("urchade/gliner_medium")

# ======================
# Train
# ======================
print("🚀 Training started...")

model.train_model(
    train_dataset=train_data,
    eval_dataset=valid_data,
    output_dir="./gliner_finetuned",
    epochs=3,
    learning_rate=2e-5,
    batch_size=2,
    eval_batch_size=2,
    logging_steps=50,
    save_steps=500,
    dataloader_num_workers=0,
    bf16=False
)

print("✅ Training finished!")
