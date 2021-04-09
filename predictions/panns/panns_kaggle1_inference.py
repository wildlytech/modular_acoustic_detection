# (https://www.apache.org/licenses/LICENSE-2.0)
#
import time
import pandas as pd
import torch
import numpy as np
from externals.models import PANNsDense121Att
import librosa
import argparse
from model_configs.panns import test_config as config
from tqdm import trange


# SEED STUFF
def seed_everything():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)


PERIOD = config["PERIOD"]
SR = config["SR"]
vote_lim = config["vote_lim"]
TTA = config["TTA"]
BIRD_CODE = config["BIRD_CODES"]
INV_BIRD_CODE = config["INV_BIRD_CODE"]
MODEL_DETAILS = config["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(ModelClass: object, config: dict, weights_path: str):
    model = ModelClass(**config)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def prediction_for_clip(clip: np.ndarray,
                        model,
                        threshold,
                        clip_threshold):
    audios = []
    y = clip.astype(np.float32)

    start = 0
    end = PERIOD * SR
    while True:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end
        end += PERIOD * SR
        audios.append(y_batch)

    array = np.asarray(audios)
    tensors = torch.from_numpy(array)

    model.eval()
    estimated_event_list = []
    global_time = 0.0
    global_cc = []
    for image in tensors:
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.expand(image.shape[0], TTA, image.shape[2])
        image = image.to(device)

        with torch.no_grad():
            _, prediction = model((image, None))
            framewise_outputs = prediction["framewise_output"].detach(
            ).cpu().numpy()[0].mean(axis=0)
            clipwise_outputs = prediction["clipwise_output"].detach(
            ).cpu().numpy()[0].mean(axis=0)

        thresholded = framewise_outputs >= threshold

        clip_thresholded = clipwise_outputs >= clip_threshold
        clip_indices = np.argwhere(clip_thresholded).reshape(-1)
        clip_codes = []
        for ci in clip_indices:
            clip_codes.append(INV_BIRD_CODE[ci])

        global_cc = list(set(global_cc) | set(clip_codes))

        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (  # noqa W504
                            detected[tail_idx + 1] -
                            detected[tail_idx] != 1):
                        onset = 0.01 * detected[
                            head_idx] + global_time
                        offset = 0.01 * detected[
                            tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[
                                         onset_idx:offset_idx, target_idx].max()  # noqa E126
                        mean_confidence = framewise_outputs[
                                          onset_idx:offset_idx, target_idx].mean()  # noqa E126

                        if INV_BIRD_CODE[target_idx] in clip_codes:
                            estimated_event = {
                                "ebird_code": INV_BIRD_CODE[target_idx],
                                "clip_codes": clip_codes,
                                "onset": onset,
                                "offset": offset,
                                "max_confidence": max_confidence,
                                "mean_confidence": mean_confidence
                            }
                            estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD

    prediction_df = pd.DataFrame(estimated_event_list)

    return prediction_df, global_cc


def predict(test_df, model, model_details):
    paths = test_df["filepath"].values
    labels = test_df["ebird_code"].values
    model.eval()
    predictions = {}
    predictions["filepath"] = []
    predictions["ebird_code"] = []
    acc_counter = 0
    total = 0
    pbar = trange(len(paths))
    pbar.set_description("Prediction Bar")
    for ii in pbar:
        time.sleep(0.01)
        # print("Iteration ",ii+1,"of ",len(paths))
        try:
            clip, _ = librosa.load(paths[ii], sr=SR, mono=True, res_type="kaiser_fast")
            total += 1
        except:
            print("Something wrong with the audioclip")

            continue

        pred_df, clip_codes = prediction_for_clip(clip, model, model_details["threshold"],
                                                  model_details["clip_threshold"])
        if len(clip_codes) == 0:
            clip_codes = ["NOCALL"]
        if labels[ii] in clip_codes:
            acc_counter += 1
        if (ii + 1) % 10 == 0:
            pbar.set_postfix(accuracy=acc_counter / total)
        #    print("Accuracy after ",(ii+1),"Iterations is ",(acc_counter/total))
        predictions["filepath"].append(paths[ii])
        if len(clip_codes) > 0:
            predictions["ebird_code"].append(clip_codes)
        else:
            predictions["ebird_code"].append("NOCALL")

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Makes Predictions using the Panns Model")
    parser.add_argument("-p", "--path_to_input", action="store", required=True, help="Path to the input data")
    parser.add_argument("-ps", "--path_to_save_preds", action="store", required=True,
                        help="Path for the preds to be saved")
    args = parser.parse_args()

    test_df = pd.read_csv(args.path_to_input)
    model_details = MODEL_DETAILS
    panns_model = get_model(PANNsDense121Att, model_details["config"], model_details["weights_path"])

    pred_df = predict(test_df, panns_model, model_details)
    print("Saving preds....")
    pred_df.to_csv(args.path_to_save_preds)
