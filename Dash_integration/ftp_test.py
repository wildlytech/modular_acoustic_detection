import ftplib
from ftplib import FTP
import pickle



# Logins into the remote ftp server  and checks for any file uploaded.
def ftp_data():

    #connect to the remote ftp
    ftp = FTP('ftp.dlptest.com',user='dlpuser@dlptest.com',passwd='e73jzTRTNqCN9PYAAjjn')
    ftp.cwd('/wav_file/')
    ftp.pwd()
    ex = ftp.nlst()
    with open('list_of_files.pkl','rb') as f:
        old_ex=pickle.load(f)

    #compare for the length of old number of files.
    if len(ex) > len(old_ex):
        ftp.quit()
        return (len(ex) - len(old_ex)), 1
    else :
        ftp.quit()
        return 0, 0


# Check the files in the directory
def get_files():
    
    ftp = FTP('ftp.dlptest.com',user='dlpuser@dlptest.com',passwd='e73jzTRTNqCN9PYAAjjn')
    files_list = ftp.retrlines('LIST')
    ftp.quit()
    return files_list


# Downloads the wav file if any. 
def download_files(value, flag):

    if flag ==1:
        ftp = FTP('ftp.dlptest.com',user='dlpuser@dlptest.com',passwd='e73jzTRTNqCN9PYAAjjn')
        wav_file_count = 0
        with open('downloaded_from_ftp.pkl','rb') as f:
            downloaded_files=pickle.load(f)
        ftp.cwd('/wav_file/')
        print ftp.nlst()
        ex = ftp.nlst()
        with open('list_of_files.pkl','rb') as f:
            old_ex=pickle.load(f)
        for each_file in ex:
                if each_file not in old_ex:
                    if each_file[-3:] == 'wav':
                        wav_file_count = wav_file_count + 1
                        print wav_file_count
                        with open(each_file, 'wb') as f:
                            ftp.retrbinary('RETR '+ each_file , f.write)
                        downloaded_files.append(each_file)
                        ex.remove(each_file)
                        print 'FTP DIr :', ftp.pwd()
                        ftp.delete(each_file)
                    else:
                        pass
                else:
                    pass
        with open('downloaded_from_ftp.pkl','wb') as f:
            pickle.dump(downloaded_files,f)
        with open('list_of_files.pkl','w') as f:
            pickle.dump(ex,f)
        ftp.quit()
        return wav_file_count
