import openGroupCsv
import pandas as pd

def concatenedToCsv():
    print('Starting generate concatened_dataset')
    # 0 = WEB
    # 1 = EMAIL
    # 2 = CHAT
    # 3 = STREAMING
    # 4 = FILE TRANSFER
    # 5 = VOIP
    # 6 = P2P

    #streaming
    vimeo = openGroupCsv.getAllCsvType('vimeo')
    youtube = openGroupCsv.getAllCsvType('youtube')
    spotify = openGroupCsv.getAllCsvType('spotify')
    netflix = openGroupCsv.getAllCsvType('netflix')
    facebook_video = openGroupCsv.getAllCsvType('facebook_video')

    streaming = [vimeo, youtube, spotify, netflix, facebook_video]
    concatened_streaming = pd.concat(streaming)
    concatened_streaming.Label = '3'

    #chat
    facebook_chat = openGroupCsv.getAllCsvType('facebook_chat')
    hangouts_chat = openGroupCsv.getAllCsvType('hangouts_chat')
    skype_chat = openGroupCsv.getAllCsvType('skype_chat')
    aim_chat = openGroupCsv.getAllCsvType('aim_chat')
    icq_chat = openGroupCsv.getAllCsvType('icq_chat')

    chat = [facebook_chat, hangouts_chat, skype_chat, aim_chat, icq_chat]
    concatened_chat = pd.concat(chat)
    concatened_chat.Label = '2'

    #voip
    skype_audio = openGroupCsv.getAllCsvType('skype_audio')
    facebook_audio = openGroupCsv.getAllCsvType('facebook_audio')
    hangouts_audio = openGroupCsv.getAllCsvType('hangouts_audio')
    voipbuster = openGroupCsv.getAllCsvType('voipbuster')

    voip = [skype_audio, facebook_audio, hangouts_audio, voipbuster]
    concatened_voip = pd.concat(voip)
    concatened_voip.Label = '5'

    # email
    email = openGroupCsv.getAllCsvType('email')

    email = [email]
    concatened_email = pd.concat(email)
    concatened_email.Label = '1'

    # file transfer
    skype_file = openGroupCsv.getAllCsvType('skype_file')
    ftps = openGroupCsv.getAllCsvType('ftps')
    sftp = openGroupCsv.getAllCsvType('sftp')
    scp = openGroupCsv.getAllCsvType('scp')

    file_transfer = [skype_file, ftps, sftp, scp]
    concatened_file_transfer = pd.concat(file_transfer)
    concatened_file_transfer.Label = '4'

    # P2P
    vpn_bittorrent = openGroupCsv.getAllCsvType('vpn_bittorrent')
    torrent = openGroupCsv.getAllCsvType('torrent')
    pearToPear = [vpn_bittorrent, torrent]
    concatened_pearToPear = pd.concat(pearToPear)
    concatened_pearToPear.Label = '6'

    # Garbage
    garbage = openGroupCsv.getAllCsvType('garbage')
    garbage = [garbage]
    concatened_garbage = pd.concat(garbage)
    concatened_garbage.Label = '7'

    concatened_dataframe = [concatened_streaming, concatened_voip, concatened_chat, concatened_file_transfer, concatened_email, concatened_pearToPear, concatened_garbage]
    concatened_dataset = pd.concat(concatened_dataframe)

    concatened_dataset = concatened_dataset[
        [
            'Flow Duration', 
            'Fwd IAT Total', 
            'Bwd IAT Total', 
            'Fwd IAT Min', 
            'Bwd IAT Min', 
            'Fwd IAT Max', 
            'Bwd IAT Max', 
            'Fwd IAT Mean', 
            'Bwd IAT Mean', 
            'Flow Packets/s',
            'Flow Bytes/s',
            'Flow IAT Min',
            'Flow IAT Max', 
            'Flow IAT Mean', 
            'Flow IAT Std', 
            'Active Min',
            'Active Mean', 
            'Active Max', 
            'Active Std', #importante?
            'Idle Min', # importante?
            'Idle Mean', #importante?
            'Idle Max', #importante?
            'Idle Std', 
            'Label'
        ]
    ].copy()

    print('cleaning dataset')
    concatened_dataset.dropna(inplace=True) # remove all inf and nan from datagrame
    concatened_dataset.to_csv('concatened_dataset.csv', index = False, decimal='.')
    print('generate concatened_dataset has been finished')