import requests
import os
import pandas as pd
import youtube_dl
import argparse


def download_clip(audio_id, clip_name, save_path,
                  root_link_path="https://cdn.download.ams.birds.cornell.edu/api/v1/asset/"):
    id_1 = audio_id
    audio_link = root_link_path + str(id_1)
    audio_files_path = save_path + clip_name
    ydl_opts = {'outtmpl': audio_files_path + '.mp3'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([audio_link])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads audio files from the macaulay library")
    parser.add_argument("-p", "--path_to_asset_ids", help="Path to the asset ids csv file", required=True)
    parser.add_argument("-sp", "--path_to_save_audioclips", help="Path to save audioclips", required=True)
    args = parser.parse_args()
    asset_ids = pd.read_csv(args.path_to_asset_ids)

    if args.path_to_save_audioclips[-1] != "/":
        args.path_to_save_audioclips += "/"

    if not os.path.exists(args.path_to_save_audioclips):
        os.mkdir(args.path_to_save_audioclips)

    for i in range(len(asset_ids)):
        clip_name = asset_ids["ClipName"][i]
        audio_id = asset_ids["Asset_ID"][i]
        download_clip(audio_id, clip_name, args.path_to_save_audioclips)
        if i == 2:
            break
