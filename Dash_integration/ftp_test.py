"""
Scrape the remote ftp server for any
.wav files and then download if any
"""
from ftplib import FTP
import pickle

# Logins into the remote ftp server  and checks for any file uploaded.
def ftp_data():
    """
    connect to the remote ftp
    """
    ftp = FTP('ftp.dlptest.com', user='dlpuser@dlptest.com', passwd='e73jzTRTNqCN9PYAAjjn')
    # ftp.cwd('/wav_file/')
    # ftp.pwd()
    ex = ftp.nlst()
    with open('downloaded_from_ftp.pkl', 'rb') as file_obj:
        downloaded_files = pickle.load(file_obj)
    EX = []
    for wav in ex:
        if len(wav) > 3 and ((wav[-3:] == 'wav') or (wav[-3:] == 'WAV')):
            if (ftp.size(wav))/1000 > 870:
            # print wav
                EX.append(wav)
    print "curated list:", EX

    #compare for the length of old number of files.
    if len(EX) > len(downloaded_files):
        ftp.close()
        return (len(EX) - len(downloaded_files)), 1
    else:
        ftp.close()
        return 0, 0

def get_files():
    """
    get the list of files in ftp directory
    """
    ftp = FTP('ftp.dlptest.com', user='dlpuser@dlptest.com', passwd='e73jzTRTNqCN9PYAAjjn')
    files_list = ftp.retrlines('LIST')
    ftp.quit()
    return files_list

def download_files(value, flag):
    """
    Download the wav file into local drive if uploaded into
    ftp server
    """
    if flag == 1:
        ftp = FTP('ftp.dlptest.com', user='dlpuser@dlptest.com', passwd='e73jzTRTNqCN9PYAAjjn')
        with open('downloaded_from_ftp.pkl', 'rb') as file_obj:
            downloaded_files = pickle.load(file_obj)
        wav_file_count = len(downloaded_files)
        # ftp.cwd('/wav_file/')
        # print ftp.nlst()
        ex = ftp.nlst()
        EX = []
        # ex = [wav_file for wav_file in ex if wav_file[-3:]=='wav' or 'WAV']
        for wav in ex:
            if len(wav) > 3 and ((wav[-3:] == 'wav') or (wav[-3:] == 'WAV')):
                if ftp.size(wav)/1000 > 870:
                    EX.append(wav)
        # with open('list_of_files.pkl', 'rb') as file_obj:
        #     old_ex = pickle.load(file_obj)
        for each_file in EX:
            if each_file not in downloaded_files:
                if each_file[-3:] == 'wav' or 'WAV':
                    try:
                        wav_file_count = wav_file_count + 1
                        print "Download count:", wav_file_count
                        with open(each_file, 'wb') as file_obj:
                            ftp.retrbinary('RETR '+ each_file, file_obj.write)
                        downloaded_files.append(each_file)
                        # ex.remove(each_file)
                        # print 'FTP DIr :', ftp.pwd()
                        # ftp.delete(each_file)
                    except:
                        "print File Error"
                else:
                    pass
            else:
                pass
        with open('downloaded_from_ftp.pkl', 'wb') as file_obj:
            pickle.dump(downloaded_files, file_obj)
        with open('file_count.pkl', 'w') as file_obj:
            pickle.dump(len(downloaded_files), file_obj)
        ftp.close()
        return wav_file_count

