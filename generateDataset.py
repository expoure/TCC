import openGroupCsv
import pandas as pd
from sklearn.utils import shuffle

def concatenedToCsv():
    print('Starting generate concatened_dataset')
    # 1 = EMAIL
    # 2 = CHAT
    # 3 = STREAMING
    # 4 = FILE TRANSFER
    # 5 = VOIP
    # 6 = P2P
    # 7 = GARBAGE

    #streaming
    vimeo = openGroupCsv.getAllCsvType('vimeo')
    youtube = openGroupCsv.getAllCsvType('youtube')
    spotify = openGroupCsv.getAllCsvType('spotify')
    netflix = openGroupCsv.getAllCsvType('netflix')
    facebook_video = openGroupCsv.getAllCsvType('facebook_video')

    streaming = [vimeo, youtube, spotify, netflix, facebook_video]
    concatened_streaming = pd.concat(streaming)
    concatened_streaming.Label = '3'
    concatened_streaming = shuffle(concatened_streaming)

    #chat
    facebook_chat = openGroupCsv.getAllCsvType('facebook_chat')
    hangouts_chat = openGroupCsv.getAllCsvType('hangouts_chat')
    skype_chat = openGroupCsv.getAllCsvType('skype_chat')
    aim_chat = openGroupCsv.getAllCsvType('aim_chat')
    icq_chat = openGroupCsv.getAllCsvType('icq_chat')

    chat = [facebook_chat, hangouts_chat, skype_chat, aim_chat, icq_chat]
    concatened_chat = pd.concat(chat)
    concatened_chat.Label = '2'
    concatened_chat = shuffle(concatened_chat)

    #voip
    skype_audio = openGroupCsv.getAllCsvType('skype_audio')
    facebook_audio = openGroupCsv.getAllCsvType('facebook_audio')
    hangouts_audio = openGroupCsv.getAllCsvType('hangouts_audio')
    voipbuster = openGroupCsv.getAllCsvType('voipbuster')

    voip = [skype_audio, facebook_audio, hangouts_audio, voipbuster]
    concatened_voip = pd.concat(voip)
    concatened_voip.Label = '5'
    concatened_voip = shuffle(concatened_voip)

    # email
    email = openGroupCsv.getAllCsvType('email')

    email = [email]
    concatened_email = pd.concat(email)
    concatened_email.Label = '1'
    concatened_email = shuffle(concatened_email)

    # file transfer
    skype_file = openGroupCsv.getAllCsvType('skype_file')
    ftps = openGroupCsv.getAllCsvType('ftps')
    sftp = openGroupCsv.getAllCsvType('sftp')
    scp = openGroupCsv.getAllCsvType('scp')

    file_transfer = [skype_file, ftps, sftp, scp]
    concatened_file_transfer = pd.concat(file_transfer)
    concatened_file_transfer.Label = '4'
    concatened_file_transfer = shuffle(concatened_file_transfer)

    # P2P
    vpn_bittorrent = openGroupCsv.getAllCsvType('vpn_bittorrent')
    torrent = openGroupCsv.getAllCsvType('torrent')
    pearToPear = [vpn_bittorrent, torrent]
    concatened_pearToPear = pd.concat(pearToPear)
    concatened_pearToPear.Label = '6'
    concatened_pearToPear = shuffle(concatened_pearToPear)

    # Garbage
    garbage = openGroupCsv.getAllCsvType('garbage')
    garbage = [garbage]
    concatened_garbage = pd.concat(garbage)
    concatened_garbage.Label = '7'
    concatened_garbage = shuffle(concatened_garbage)

    ## SEPARANDO PARA VALIDAÇÃO
    print('concatened_streaming', len(concatened_streaming.index))
    print('concatened_voip', len(concatened_voip.index))
    print('concatened_chat', len(concatened_chat.index))
    print('concatened_file_transfer', len(concatened_file_transfer.index))
    print('concatened_email', len(concatened_email.index))
    print('concatened_pearToPear', len(concatened_pearToPear.index))
    print('concatened_garbage', len(concatened_garbage.index))
    ##

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

    #ISSO AQUI PODE SERVIR COMO ARGUMENTAÇÃO PARA AS COLUNAS ESCOLHIDAS
    meanLabels = concatened_dataset.groupby('Label').mean().round()
    stdLabels = concatened_dataset.groupby('Label').std().round()

    print('saving statistics')
    meanLabels.to_csv(r'statistics/mean.csv', index = False, decimal='.')
    stdLabels.to_csv(r'statistics/std.csv', index = False, decimal='.')

    concatened_dataset.to_csv('concatened_dataset.csv', index = False, decimal='.')
    print('generate concatened_dataset has been finished')
