import argparse
import json
from pprint import pprint

from externals.models import CustomPanns
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn  # noqa F401
import torch.optim as optim  # noqa F401
from sklearn.model_selection import train_test_split
import tqdm

# SEED EVERYTHING
torch.manual_seed(42)
np.random.seed(42)


def get_model(config):
    model_config = config["MODEL_CONFIG"]

    model = CustomPanns(
        **model_config
    )
    return model


def df_checker(df_path, SR=32000):
    df = pd.read_csv(df_path)
    new_df_dict = {}
    new_df_dict["filepath"] = []
    new_df_dict["bird_name"] = []
    new_df_dict["ebird_code"] = []
    for ii, filename in enumerate(df["filepath"]):
        print("Iteration: ", ii + 1)

        try:

            audio, _ = librosa.load(filename, sr=SR, mono=True, res_type="kaiser_fast")
            new_df_dict["filepath"].append(filename)
            new_df_dict["bird_name"].append(df["bird_name"][ii])
            new_df_dict["ebird_code"].append(df["ebird_code"][ii])

        except Exception as e:
            print(e)
            inp = input("Enter y to break")
            if inp == "y":
                break
            print(new_df_dict)

    return pd.DataFrame(new_df_dict)


class Audio_Dataset(Dataset):
    def __init__(self, csv, enc_dict, num_classes):
        self.csv = csv
        self.enc_dict = enc_dict
        self.num_classes = num_classes

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.csv["filepath"][idx]

        label = self.csv["ebird_code"][idx]
        zs = np.zeros((1, self.num_classes))
        zs[:, self.enc_dict[label]] = 1
        clip, _ = librosa.load(path, sr=SR, mono=True, res_type="kaiser_fast")
        array = clip.astype(np.float32)
        if len(array) < 960000:
            array = np.pad(array, (0, 960000 - len(array)), mode='constant', constant_values=0)
        # print(path)
        # print(len(array))
        tensors = torch.from_numpy(array)

        label = torch.Tensor(zs)
        return tensors, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fintuner for PANNs model")
    parser.add_argument("-train", "--train_path", required=True, help="Path to train dataframe (csv file)")
    parser.add_argument("-test", "--test_path", required=True, help="Path to test dataframe (csv file)")
    parser.add_argument("-p2p", "--path_to_prediction", required=True, help="Path to save predictions (csv file)")
    parser.add_argument("-cfg", "-path2config", required=True, help="Path to the json config file")
    args = parser.parse_args()
    data_csv = pd.read_csv(args.train_path)

    test_data_csv = pd.read_csv(args.test_path)
    with open(args.cfg, "r") as f:
        config = json.load(f)

    PERIOD = config["PERIOD"]
    SR = config["SR"]
    vote_lim = config["vote_lim"]
    TTA = config["TTA"]
    BATCH_SIZE = config["BATCH_SIZE"]
    BIRD_CODE = config["BIRD_CODE"]
    INV_BIRD_CODE = {v: k for k, v in config["BIRD_CODE"].items()}

    model = get_model(config)

    df_train, df_val = train_test_split(data_csv, test_size=0.2, random_state=42)
    df_train.reset_index(inplace=True)
    df_val.reset_index(inplace=True)
    train_dset = Audio_Dataset(df_train, BIRD_CODE, 268)
    val_dset = Audio_Dataset(df_val, BIRD_CODE, 268)
    test_dset = Audio_Dataset(test_data_csv, BIRD_CODE, 268)

    train_dataloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_dset, batch_size=BATCH_SIZE, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    EPOCHS = config["EPOCHS"]
    criterion = eval(config["LOSS"])
    optimizer = eval(config["OPTIM"])
    adam = optimizer(model.parameters(), lr=config["LR"])

    master_bar = tqdm.trange(EPOCHS, unit="Epochs")

    for epoch in master_bar:

        acc_score = 0.0
        model.train()
        pbar_t = tqdm.tqdm(train_dataloader, unit="batch", leave=False)
        pbar_t.set_description(f"Epoch: {epoch}")
        for ii, (x, y) in enumerate(pbar_t):
            adam.zero_grad()
            image = x.unsqueeze(1)

            image = image.to(device)
            y = y.to(device)

            out = model((image, None))
            try:
                clipwise_op = torch.clamp(out["combined_output"], 0, 1)

                preds = np.argmax(clipwise_op.detach().cpu().squeeze(1).numpy(), axis=1)

                target = np.argmax(y.detach().cpu().squeeze(1).numpy(), axis=1)

                acc_score += (np.sum(target == preds)) / BATCH_SIZE
                loss = criterion(clipwise_op[:, :, -4:], y[:, :, -4:])

                acc_score = acc_score / (ii + 1)
                pbar_t.set_postfix(loss=loss.item(), accuracy=acc_score)
            except Exception as e:
                print(e)

            loss.backward()
            adam.step()

        model.eval()
        val_acc = 0.0
        val_preds = []
        val_targets = []
        pbar_v = tqdm.tqdm(val_dataloader, unit="batch", leave=False)
        pbar_v.set_description("Validation")
        for ii, (x, y) in enumerate(pbar_v):
            image = x.unsqueeze(1)

            image = image.to(device)
            y = y.to(device)
            with torch.no_grad():
                out = model((image, None))
            clipwise_op = torch.clamp(out["combined_output"], 0, 1)

            preds = np.argmax(clipwise_op.detach().cpu().squeeze(1).numpy(), axis=1)

            target = np.argmax(y.detach().cpu().squeeze(1).numpy(), axis=1)
            val_preds.extend([INV_BIRD_CODE[i] for i in preds.tolist()])
            val_targets.extend([INV_BIRD_CODE[i] for i in target.tolist()])
            val_acc += (np.sum(target == preds)) / BATCH_SIZE
            val_loss = criterion(clipwise_op, y)
            val_acc = val_acc / (ii + 1)

            pbar_v.set_postfix(loss=val_loss.item(), accuracy=val_acc)

        pred_df = {}
        pred_df["preds"] = val_preds
        pred_df["target"] = val_targets
        pd.DataFrame(pred_df).to_csv(args.path_to_prediction)
        # print("Val Accuracy: ",val_acc/len(val_dataloader))
        master_bar.set_postfix(train_loss=loss.item(), val_loss=val_loss.item(),
                               train_acc=acc_score, val_acc=val_acc)

    # TESTING
    # print("Testing....")
    test_bar = tqdm.tqdm(test_dataloader, unit="batch")
    test_bar.set_description("Testing")
    model.eval()
    test_acc = 0.0
    test_preds = []
    test_targets = []
    clip_threshold = 0.2
    for ii, (x, y) in enumerate(test_bar):
        image = x.unsqueeze(1)

        image = image.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model((image, None))

        clip_op = out["clipwise_output"].detach().cpu().numpy().mean(axis=1)
        backbone_op = out["combined_output"].detach().cpu().numpy().mean(axis=1)

        clip_indices = np.argsort(-backbone_op, axis=1)
        clip_indices = clip_indices[:, :1]

        clip_codes = []
        for ci_arr in clip_indices:
            inv_codes = []
            for ci in ci_arr:
                inv_codes.append(INV_BIRD_CODE[ci])
            clip_codes.append(inv_codes)
        test_preds.extend(clip_codes)

        target = np.argmax(y.detach().cpu().squeeze(1).numpy(), axis=1)

        test_targets.extend([INV_BIRD_CODE[i] for i in target.tolist()])

    print(test_preds[:10])
    pred_df = {}
    print("Preds: ", len(test_preds))
    print("Targets: ", len(test_targets))
    pred_df["preds"] = test_preds
    pred_df["target"] = test_targets
    pd.DataFrame(pred_df).to_csv(args.path_to_prediction)
