
# Set Finetuned model path
modelPath = "Babelscape/rebel-large"
modelCheckpointPath = "/content/drive/MyDrive/Colab Notebooks/timexes_thesis/model/"
# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/timexes_thesis/data/"
encodedTrainPath = dataDir + "encodedtrain/"
encodedtestPath = dataDir + "encodedtest/"

# LOAD DATASETS
from datasets import load_from_disk
trainDSTK = load_from_disk(encodedTrainPath)
trainDSTK = trainDSTK['train']
testDSTK = load_from_disk(encodedtestPath)
testDSTK = testDSTK['train']

# Load model and tokenizer from checkpoint if base model not already loaded
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)

# Set model to CUDA processor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set training arguments
batchSize = 8

train_args = Seq2SeqTrainingArguments(
  output_dir=modelFTPath, 
  overwrite_output_dir=True, 
  evaluation_strategy="epoch",
  per_device_train_batch_size=batchSize, # batch size per device during training
  per_device_eval_batch_size=batchSize, # batch size for evaluation
  warmup_steps=500, # number of warmup steps for learning rate scheduler
  weight_decay=0.01, # strength of weight decay
  num_train_epochs=10,
  save_total_limit=1)

# Set Data Collator for batching
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Set Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=False)

# Set Trainer
trainer = Seq2SeqTrainer(
    model = model,
    args = train_args,
    data_collator = data_collator,
    train_dataset = trainDSTK,
    eval_dataset = testDSTK,
    tokenizer = tokenizer,
    optimizers = (optimizer, scheduler))

# Fine-tune model
trainer.train()

# Save model checkpoint
trainer.save_model(modelCheckpointPath)
