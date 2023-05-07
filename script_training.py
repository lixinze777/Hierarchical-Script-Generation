import os
import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console


console = Console(record=True)

def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Perplexity", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

device = 'cuda' if cuda.is_available() else 'cpu'

class DataSet(Dataset):

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer):

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 1000 == 0:
            print(loss)
            ppl = torch.exp(loss)
            training_logger.add_row(str(epoch), str(_), str(ppl))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader):

  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
              num_beams=2,
              repetition_penalty=10.0, 
              length_penalty=0.01, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

          if _%100==0:
              console.print(f'Completed {_}')
              lm_labels = y[:, 1:].clone().detach()
              lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
              outputs = model(generated_ids, labels=lm_labels)
              neg_log_likelihood = outputs[0]
              ppl = torch.exp(neg_log_likelihood)
              console.print(f'Perplexity {ppl}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def T5Trainer(dataframe, source_text, target_text, model_params, output_dir="../outputs/t5_0608"):

    torch.manual_seed(model_params["SEED"])
    np.random.seed(model_params["SEED"])
    torch.backends.cudnn.deterministic = True

    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens(['<section>'])

    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    console.log(f"[Data]: Reading data...\n")

    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Defining the train size. So 90% of the data will be used for training and the rest for Testing.
    # From the 90% of data, 5% will be held for validation
    train_size = 0.9
    dev_size = 0.05
    train_dev_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    test_dataset = dataframe.drop(train_dev_dataset.index).reset_index(drop=True)
    train_dev_dataset = train_dev_dataset.reset_index(drop=True)

    train_dataset = train_dev_dataset.sample(frac=(1-dev_size), random_state=model_params["SEED"])
    dev_dataset = train_dev_dataset.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)   

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"DEV Dataset: {dev_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    training_set = DataSet(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = DataSet(
        dev_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    titles = val_set.source_text

    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    console.log(f"[Initiating validation on dev dataset]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Titles":titles, "Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Inference Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Evaluation] Generation on dev data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__ == "__main__":

    model_params = {
        "MODEL": "t5-base",
        "TRAIN_BATCH_SIZE": 16,
        "VALID_BATCH_SIZE": 16,
        "TRAIN_EPOCHS": 3, 
        "VAL_EPOCHS": 1, 
        "LEARNING_RATE": 1e-4, 
        "MAX_SOURCE_TEXT_LENGTH": 64,
        "MAX_TARGET_TEXT_LENGTH": 512,
        "SEED": 42, 
    }

    path = "../data/wikihow/wikiwithPredictedSection.csv"
    df = pd.read_csv(path)
    df["title"] = "ask_question: " + df["title"]
    df["method"] = "answer: "+df["method"]

    T5Trainer(
        dataframe=df,
        source_text="title",
        target_text="method",
        model_params=model_params,
        output_dir="../outputs/wikihow/dev",
    )