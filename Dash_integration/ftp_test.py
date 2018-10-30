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
    ftp.cwd('/wav_file/')
    ftp.pwd()
    ex = ftp.nlst()
    with open('list_of_files.pkl', 'rb') as file_obj:
        old_ex = pickle.load(file_obj)

    #compare for the length of old number of files.
    if len(ex) > len(old_ex):
        ftp.quit()
        return (len(ex) - len(old_ex)), 1
    else:
        ftp.quit()
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
    Download the wav file if uploaded into
    ftp server
    """
    if flag == 1:
        ftp = FTP('ftp.dlptest.com', user='dlpuser@dlptest.com', passwd='e73jzTRTNqCN9PYAAjjn')
        wav_file_count = 0
        with open('downloaded_from_ftp.pkl', 'rb') as file_obj:
            downloaded_files = pickle.load(file_obj)
        ftp.cwd('/wav_file/')
        print ftp.nlst()
        ex = ftp.nlst()
        with open('list_of_files.pkl', 'rb') as file_obj:
            old_ex = pickle.load(file_obj)
        for each_file in ex:
            if each_file not in old_ex:
                if each_file[-3:] == 'wav':
                    wav_file_count = wav_file_count + 1
                    print wav_file_count
                    with open(each_file, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+ each_file, file_obj.write)
                    downloaded_files.append(each_file)
                    ex.remove(each_file)
                    print 'FTP DIr :', ftp.pwd()
                    ftp.delete(each_file)
                else:
                    pass
            else:
                pass
        with open('downloaded_from_ftp.pkl', 'wb') as file_obj:
            pickle.dump(downloaded_files, file_obj)
        with open('list_of_files.pkl', 'w') as file_obj:
            pickle.dump(ex, file_obj)
        ftp.quit()
        return wav_file_count
