import pandas
import requests
import argparse

import youtube_dl


def download_clip(audio_id, clip_name, save_path,
                  root_link_path="https://cdn.download.ams.birds.cornell.edu/api/v1/asset/"):
    id_1 = audio_id
    audio_link = root_link_path + str(id_1)
    audio_files_path = save_path + clip_name
    ydl_opts = {'outtmpl': audio_files_path + '.mp3'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([audio_link])


def read_codes(path):
    with open(path, "r") as f:
        codes = f.readlines()

    codes = [code.rstrip("\n") for code in codes]
    return codes


def search_records(url, l_limit=0, u_limit=100000):
    while (True):
        mid = (l_limit + u_limit) // 2
        mid_url = url + str(mid)
        page = requests.get(mid_url)
        try:
            res = page.json()["results"]  # noqa F841

            if mid == l_limit:
                break
            else:
                l_limit = mid
        except:
            u_limit = mid
        print("The page has between ", l_limit, "results and ", u_limit, " results")
    print("The page has ", mid, " results")
    return mid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads audioclips from macaulay library")
    parser.add_argument("-p", "--path_to_taxon_codes", help="Path to where text file containing taxon codes is saved",
                        required=True)
    parser.add_argument("-sp", "--save_path", help="Path to save the audio clips", default="macaulay_audio/")
    args = parser.parse_args()

    codes = read_codes(args.path_to_taxon_codes)
    df_dict = {}
    asset_ids = []
    bird_names = []
    for code in codes:

        print("Operating code: ", code)
        url = "https://search.macaulaylibrary.org/catalog.json?searchField=species&taxonCode=" + code + "&count="
        print("Searching for number of results..")
        # mid = search_records(url)
        mid = 100
        final_url = url + str(mid)
        page = requests.get(final_url)
        if len(page.json()["results"]["content"]) == 0:
            print("No results found for ", page.json()["searchRequestForm"]["q"])
        else:
            num_recs = len(page.json()["results"]["content"])
            asset_ids.extend([el["assetId"] for el in page.json()["results"]["content"]])
            bird_names.extend([page.json()["searchRequestForm"]["q"] for i in range(num_recs)])
        print("A: ", asset_ids)
        print(len(asset_ids))
        break

    df_dict["Asset_ID"] = asset_ids
    df_dict["bird_names"] = bird_names
    df = pandas.DataFrame(df_dict)
    for i in range(len(df)):
        audio_id = df["Asset_ID"][i]
        clip_name = "Macaulay_" + str(audio_id)
        download_clip(audio_id, clip_name, args.save_path)
