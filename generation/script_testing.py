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
    Column("Loss", justify="center"),
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
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

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


def inference(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
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

    # Defining the train size. So 90% of the data will be used for training and the rest for validation.
    train_size = 0.9
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    test_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    display_df(test_dataset.head(2))

    console.print(f"TEST Dataset: {test_dataset.shape}\n")
    test_set = DataSet(
        test_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    titles = test_set.source_text

    test_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    test_loader = DataLoader(test_set, **test_params)

    console.log(f"[Initiating Testing ON TEST SET]...\n")
    for epoch in range(model_params["TEST_EPOCHS"]):
        predictions, actuals = inference(epoch, tokenizer, model, device, test_loader)
        final_df = pd.DataFrame({"Titles":titles, "Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions0116.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Testing Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions0116.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__ == "__main__":

    model_params = {
        "MODEL": "../outputs/wikihow/model_files",
        "TEST_BATCH_SIZE": 16,
        "TEST_EPOCHS": 1,
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
        output_dir="../outputs/wikihow",
    )
