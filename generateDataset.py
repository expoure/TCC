import openGroupCsv
import pandas as pd
from sklearn.utils import shuffle

columns = [
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

def percentage(percent, whole):
  return int((percent * whole) / 100.0)

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
    temp_concatened_streaming = shuffle(concatened_streaming)

    streaming_len = len(temp_concatened_streaming.index) - 1
    streaming_quantity_validation = percentage(20, len(temp_concatened_streaming.index))

    validate_streaming = pd.DataFrame(temp_concatened_streaming.take([0]))
    concatened_streaming = pd.DataFrame(temp_concatened_streaming.take([0]))
    for row in range(1, streaming_len):
        if (row > (streaming_len - streaming_quantity_validation)):
            validate_streaming = validate_streaming.append(temp_concatened_streaming.take([row]))
        else:
            concatened_streaming = validate_streaming.append(temp_concatened_streaming.take([row]))

    validate_streaming.dropna(inplace=True)
    validate_streaming.to_csv('data/validate_streaming.csv', index = False, decimal='.')

    #chat
    facebook_chat = openGroupCsv.getAllCsvType('facebook_chat')
    hangouts_chat = openGroupCsv.getAllCsvType('hangouts_chat')
    skype_chat = openGroupCsv.getAllCsvType('skype_chat')
    aim_chat = openGroupCsv.getAllCsvType('aim_chat')
    icq_chat = openGroupCsv.getAllCsvType('icq_chat')

    chat = [facebook_chat, hangouts_chat, skype_chat, aim_chat, icq_chat]
    concatened_chat = pd.concat(chat)
    concatened_chat.Label = '2'
    temp_concatened_chat = shuffle(concatened_chat)

    chat_len = len(temp_concatened_chat.index) - 1
    chat_quantity_validation = percentage(20, len(temp_concatened_chat.index))

    validate_chat = pd.DataFrame(temp_concatened_chat.take([0]))
    validate_chat = pd.DataFrame(temp_concatened_chat.take([0]))
    for row in range(1, chat_len):
        if (row > (chat_len - chat_quantity_validation)):
            validate_chat = validate_chat.append(temp_concatened_chat.take([row]))
        else:
            concatened_chat = validate_chat.append(temp_concatened_chat.take([row]))
    validate_chat.dropna(inplace=True)
    validate_chat.to_csv('data/validate_chat.csv', index = False, decimal='.')

    #voip
    skype_audio = openGroupCsv.getAllCsvType('skype_audio')
    facebook_audio = openGroupCsv.getAllCsvType('facebook_audio')
    hangouts_audio = openGroupCsv.getAllCsvType('hangouts_audio')
    voipbuster = openGroupCsv.getAllCsvType('voipbuster')

    voip = [skype_audio, facebook_audio, hangouts_audio, voipbuster]
    concatened_voip = pd.concat(voip)
    concatened_voip.Label = '5'
    temp_concatened_voip = shuffle(concatened_voip)

    voip_len = len(temp_concatened_voip.index) - 1
    voip_quantity_validation = percentage(20, len(temp_concatened_voip.index))

    validate_voip = pd.DataFrame(temp_concatened_voip.take([0]))
    validate_voip = pd.DataFrame(temp_concatened_voip.take([0]))
    for row in range(1, voip_len):
        if (row > (voip_len - voip_quantity_validation)):
            validate_voip = validate_voip.append(temp_concatened_voip.take([row]))
        else:
            concatened_voip = validate_voip.append(temp_concatened_voip.take([row]))
    validate_voip.dropna(inplace=True)
    validate_voip.to_csv('data/validate_voip.csv', index = False, decimal='.')

    # email
    email = openGroupCsv.getAllCsvType('email')

    email = [email]
    concatened_email = pd.concat(email)
    concatened_email.Label = '1'
    temp_concatened_email = shuffle(concatened_email)

    email_len = len(temp_concatened_email.index) - 1
    email_quantity_validation = percentage(20, len(temp_concatened_email.index))

    validate_email = pd.DataFrame(temp_concatened_email.take([0]))
    validate_email = pd.DataFrame(temp_concatened_email.take([0]))
    for row in range(1, email_len):
        if (row > (email_len - email_quantity_validation)):
            validate_email = validate_email.append(temp_concatened_email.take([row]))
        else:
            concatened_email = validate_email.append(temp_concatened_email.take([row]))
    validate_email.dropna(inplace=True)
    validate_email.to_csv('data/validate_email.csv', index = False, decimal='.')

    # file transfer
    skype_file = openGroupCsv.getAllCsvType('skype_file')
    ftps = openGroupCsv.getAllCsvType('ftps')
    sftp = openGroupCsv.getAllCsvType('sftp')
    scp = openGroupCsv.getAllCsvType('scp')

    file_transfer = [skype_file, ftps, sftp, scp]
    concatened_file_transfer = pd.concat(file_transfer)
    concatened_file_transfer.Label = '4'
    temp_concatened_file_transfer = shuffle(concatened_file_transfer)

    file_transfer_len = len(temp_concatened_file_transfer.index) - 1
    file_transfer_quantity_validation = percentage(20, len(temp_concatened_file_transfer.index))

    validate_file_transfer = pd.DataFrame(temp_concatened_file_transfer.take([0]))
    validate_file_transfer = pd.DataFrame(temp_concatened_file_transfer.take([0]))
    for row in range(1, file_transfer_len):
        if (row > (file_transfer_len - file_transfer_quantity_validation)):
            validate_file_transfer = validate_file_transfer.append(temp_concatened_file_transfer.take([row]))
        else:
            concatened_file_transfer = validate_file_transfer.append(temp_concatened_file_transfer.take([row]))
    validate_file_transfer.dropna(inplace=True)
    validate_file_transfer.to_csv('data/validate_file_transfer.csv', index = False, decimal='.')

    # P2P
    vpn_bittorrent = openGroupCsv.getAllCsvType('vpn_bittorrent')
    torrent = openGroupCsv.getAllCsvType('torrent')
    pearToPear = [vpn_bittorrent, torrent]
    concatened_pearToPear = pd.concat(pearToPear)
    concatened_pearToPear.Label = '6'
    temp_concatened_pearToPear = shuffle(concatened_pearToPear)

    p2p_len = len(temp_concatened_pearToPear.index) - 1
    p2p_quantity_validation = percentage(20, len(temp_concatened_pearToPear.index))

    validate_p2p = pd.DataFrame(temp_concatened_pearToPear.take([0]))
    validate_p2p = pd.DataFrame(temp_concatened_pearToPear.take([0]))
    for row in range(1, p2p_len):
        if (row > (p2p_len - p2p_quantity_validation)):
            validate_p2p = validate_p2p.append(temp_concatened_pearToPear.take([row]))
        else:
            concatened_p2p = validate_p2p.append(temp_concatened_pearToPear.take([row]))
    validate_p2p.dropna(inplace=True)
    validate_p2p.to_csv('data/validate_p2p.csv', index = False, decimal='.')

    # Garbage
    garbage = openGroupCsv.getAllCsvType('garbage')
    garbage = [garbage]
    concatened_garbage = pd.concat(garbage)
    concatened_garbage.Label = '7'
    temp_concatened_garbage = shuffle(concatened_garbage)

    garbage_len = len(temp_concatened_garbage.index) - 1
    garbage_quantity_validation = percentage(20, len(temp_concatened_garbage.index))

    validate_garbage = pd.DataFrame(temp_concatened_garbage.take([0]))
    validate_garbage = pd.DataFrame(temp_concatened_garbage.take([0]))
    for row in range(1, garbage_len):
        if (row > (garbage_len - garbage_quantity_validation)):
            validate_garbage = validate_garbage.append(temp_concatened_garbage.take([row]))
        else:
            concatened_garbage = validate_garbage.append(temp_concatened_garbage.take([row]))
    validate_garbage.dropna(inplace=True)
    validate_garbage.to_csv('data/validate_garbage.csv', index = False, decimal='.')

    ## SEPARANDO PARA VALIDAÇÃO


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
