import os
import sys
from mega import Mega

if __name__ == "__main__":
    mega = Mega()
    m = mega.login()

    datasets_dir = sys.argv[1]
    os.makedirs(datasets_dir, exist_ok=True)

    # Electricity
    print('Downloading Electricity dataset...')
    m.download_url('https://mega.nz/file/MaUxSTzI#SbVLiltqBzf9wa9C4zoE_goMYZY_rikTXFGodynSTmo', datasets_dir)
    m.download_url('https://mega.nz/file/lCElEBRA#RonhMz5aO4wDevhXTb6cqg_yjWegs0lRiY6yY2SuAvM', datasets_dir)
    print('Electricity dataset downloaded')
    
    # ETTm1
    print('Downloading ETTm1 dataset...')
    m.download_url('https://mega.nz/file/8LlFRKKA#1B15mP2Nu8bg2UXBvDvlb2wQtrhgqmr7NBgtgh9GynQ', datasets_dir)
    m.download_url('https://mega.nz/file/lbVHTKpJ#jIHxfQlx-WWW-tSX7juHsyYYyMrfEiQdGD17jvVINYk', datasets_dir)
    print('ETTm1 dataset downloaded')
    
    # Mujoco
    print('Downloading Mujoco dataset...')
    m.download_url('https://mega.nz/file/YKtwkSjZ#u_yJy3KNZyUfGYCFzwNNU7Nzar232-5r_jrCLSNmq50', datasets_dir)
    m.download_url('https://mega.nz/file/QXsBgQoS#CjbChp35YZ_sKIjrB0YLyuayfCWHSBuKJR2AZnZJO8k', datasets_dir)
    print('Mujoco dataset downloading')
    
    # PTB-XL
    print('Downloading PTB-XL dataset 248...')
    m.download_url('https://mega.nz/file/wKFmjbQY#9pQCnYAV282xJlkuJ1cAsgklHQj8toYCFylGZl5DC-w', datasets_dir)
    m.download_url('https://mega.nz/file/UDkByCLY#SwL3NyAhtkJKbvn6PEosnN9mTOZb4yT0PHaW6fMQU3k', datasets_dir)
    m.download_url('https://mega.nz/file/IOMzTIiI#w3wu0SNnelnDaoyn3cZXyvqTXLlf587SPWsyQOWYESc', datasets_dir)
    
    print('Downloading PTB-XL dataset 1000....')
    m.download_url('https://mega.nz/file/ZCtkFbZT#U4lDsYUZx_oiLX8QQVZLg4_bTFEM_xR2Xn3YCoTSPFM', datasets_dir)
    m.download_url('https://mega.nz/file/IS9Q1ZaT#3syB-EH_s0rI3riTzBnY9CtkGRxiUcBgqwsvig1uEQs', datasets_dir)
    m.download_url('https://mega.nz/file/oHUlUI4J#T4F3n1UdV0yZwZ1i9NZE_Vz-nHr5uxeYC49oh9jLqo4', datasets_dir)
    
    print('PTB-XL datasets downloaded')