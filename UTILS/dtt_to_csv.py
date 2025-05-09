import pandas as pd
import csv
import numpy as np
import time
from datetime import datetime
import os
import os.path
import re
from collections import OrderedDict

folder = "DTT_TO_CSV"

if os.path.isdir(folder):
    pass
else:
    os.mkdir(folder)
    os.mkdir(os.path.join(os.getcwd(), folder, "seed"))
    os.mkdir(os.path.join(os.getcwd(), folder, "target"))

filename_rls = 'Vesuvio_2015_ok.rls'
filename_dtt = 'Vesuvio_2015_ok.dtt'

# -------Reading rls--------

# Lista per i valori della quarta colonna
realization = []

# Apri il file in modalità lettura
with open(filename_rls, 'r', encoding='latin-1') as file:
    # Leggi il contenuto del file riga per riga
    for riga in file:
        # Dividi la riga in colonne utilizzando uno spazio come delimitatore
        colonne = riga.strip().split()

        # Verifica se ci sono abbastanza colonne nella riga
        if len(colonne) >= 4:
            # Aggiungi il valore della quarta colonna alla lista dei valori
            realization.append(float(colonne[2]))

# Stampa la lista dei valori della quarta colonna
print('realizations', realization)

# -------Reading dtt--------
# Apri il file in modalità lettura
with open(filename_dtt, 'r', encoding='latin-1') as file:

    # Leggi la prima riga
    prima_riga = file.readline().strip().split()

    # Estrai gli ultimi 3 numeri dalla prima riga
    numeri_percentili = [int(num) for num in prima_riga[-3:]]

    # Definisci la lista di percentili
    percentiles = [f'{num}%ile' for num in numeri_percentili]

    print('percentiles', percentiles)

    # Salta la prima riga
    for _ in range(1):
        next(file)

# Apri il file in modalità lettura
with open(filename_dtt, 'r', encoding='latin-1') as file:
    # Crea un lettore CSV
    lettore_csv = csv.reader(file)

    # Leggi la prima riga come header, poi leggi fino all'ultima domanda del
    # primo esperto per ottenere il testo delle domande e i nomi - mod AT
    idx_expert = []
    expert_name = []
    idx_q = []
    type_and_idx_q = []
    scale = []
    perc1 = []
    perc2 = []
    perc3 = []
    names_f = []
    qst_txt_f = []

    for _ in range(1):
        riga = next(lettore_csv)

    for riga in lettore_csv:
        ie_1 = riga[0][0:5]
        idx_expert.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", ie_1)))
        en_1 = riga[0][6:14]
        expert_name.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", en_1)))
        iq_1 = riga[0][15:19]
        idx_q.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", iq_1)))
        SQ_1 = riga[0][21:34]
        type_and_idx_q.append(" ".join(
            re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", SQ_1)))
        scal1 = riga[0][34:38]
        scale.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", scal1)))
        perc1_1 = riga[0][38:53]
        perc1.append(perc1_1)
        perc2_1 = riga[0][54:68]
        perc2.append(perc2_1)
        perc3_1 = riga[0][69:83]
        perc3.append(perc3_1)
        FN_1 = riga[0][260:308]
        names_f.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", FN_1)))
        LQ_1 = riga[0][84:259]
        qst_txt_f.append(" ".join(re.findall(r"[\wÀ-ÿ]+|\d+|[^\w\s]", LQ_1)))
    result = list(OrderedDict.fromkeys(type_and_idx_q))
    for i in range(len(result)):
        if qst_txt_f[i] == '':
            qst_txt_f[i] = result[i]

    header = ["idx_expert", "expert_name", "idx_q", "type_and_idx_q", "scale"
              ] + percentiles
    dati = list(
        zip(idx_expert, expert_name, idx_q, type_and_idx_q, scale, perc1,
            perc2, perc3))
    df = pd.DataFrame(dati, columns=header)

    names = list(filter(None, names_f))
    qst_txt = list(filter(None, qst_txt_f))

# Crea un DataFrame utilizzando l'header e i dati

df = df.replace('-9.99500E+0002', '')
df = df.replace('-9.99600E+0002', '')
print(df)

idx_experts = np.unique(np.array(df['idx_expert'], dtype=int))
n_experts = idx_experts.shape[0]

header_quest = [
    'IDX', 'LABEL', 'SHORT Q', 'LONG Q_ENG', 'UNITS', 'SCALE', 'MINVAL',
    'MAXVAL', 'REALIZATION', 'QUEST_TYPE', 'IDXMIN', 'IDXMAX', 'SUM50',
    'PARENT', 'IMAGE'
]
data_quest = []

for count, idx in enumerate(idx_experts):

    print('Seed Questions for Exp', idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    name_parts = names[count].split()

    columns.append('First Name')
    data.append(name_parts[0])

    columns.append('Last Name')
    if len(name_parts) == 1:
        data.append('N/A')
    elif len(name_parts) == 2:
        data.append(name_parts[1])
    elif len(name_parts) > 2:
        words = name_parts[1:]
        combined = " ".join(words)
        data.append(combined)
    columns.append('Email address')
    data.append(names[count].rsplit(' ', 1)[0] + '@mail.com')

    columns.append('Group(s)')
    data.append('0')
    print(names[count].rsplit(' ', 1)[0])

    n_SQ = 0
    k = 0

    for index, row in rslt_df.iterrows():

        row_sq = []

        if realization[k] != -999.5 and realization[k] != -999.6:
            n_SQ += 1
            k += 1
            # seed_idx = re.findall(r'\d+', row['type_and_idx_q'])[0]
            seed_idx = k

            columns.append(row['type_and_idx_q'] + ' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q'] + ' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q'] + ' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:
                row_sq.append(seed_idx)
                row_sq.append(row['type_and_idx_q'])
                row_sq.append(row['type_and_idx_q'])
                row_sq.append(qst_txt[index])
                row_sq.append('[]')
                row_sq.append(str(row['scale']).lower())
                row_sq.append(0)
                row_sq.append('inf')
                row_sq.append(str(realization[int(seed_idx) - 1]))
                row_sq.append('seed')
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(-1)
                row_sq.append('')

                data_quest.append(row_sq)
        else:
            k += 1

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Write the merged data to a new CSV file
    if all(data) is True:
        new_filename = './DTT_TO_CSV/seed/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row
    else:
        print("Warning: Exp %s has not given answers to one or more seed, "
              "saving his/her answers in a separate folder" %
              names[count].rsplit(' ', 1)[0])
        if os.path.isdir('./DTT_TO_CSV/seed_missing'):
            pass
        else:
            os.mkdir('./DTT_TO_CSV/seed_missing')
        new_filename = './DTT_TO_CSV/seed_missing/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row

    time.sleep(2)

for count, idx in enumerate(idx_experts):

    print('Target Questions for Exp', idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    name_parts = names[count].split()

    columns.append('First Name')
    data.append(name_parts[0])

    columns.append('Last Name')
    if len(name_parts) == 1:
        data.append('N/A')
    elif len(name_parts) == 2:
        data.append(name_parts[1])
    elif len(name_parts) > 2:
        words = name_parts[1:]
        combined = " ".join(words)
        data.append(combined)

    columns.append('Email address')
    data.append(names[count].rsplit(' ', 1)[0] + '@mail.com')

    columns.append('Group(s)')
    data.append('0')
    print(names[count].rsplit(' ', 1)[0])

    n_TQ = 0
    k = 0

    for index, row in rslt_df.iterrows():

        row_tq = []

        if realization[k] == -999.5 or realization[k] == -999.6:
            n_TQ += 1
            k += 1
            # target_idx = re.findall(r'^\D*(\d+)', row['type_and_idx_q'])[0]
            target_idx = k
            columns.append(row['type_and_idx_q'] + ' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q'] + ' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q'] + ' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:
                row_tq.append(n_TQ)
                row_tq.append(row['type_and_idx_q'])
                row_tq.append(row['type_and_idx_q'])
                row_tq.append(qst_txt[index])
                row_tq.append('[]')
                row_tq.append(str(row['scale']).lower())
                row_tq.append(0)
                row_tq.append('inf')
                row_tq.append(' ')
                row_tq.append('target')
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(-1)
                row_tq.append('')

                data_quest.append(row_tq)
        else:
            k += 1

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Write the merged data to a new CSV file
    if all(data) is True:
        new_filename = './DTT_TO_CSV/target/questionnaire_' + dt_string \
            + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row
    else:
        print("Warning: Exp %s has not given answers to one or more target,"
              "saving his/her answers in a separate folder" %
              names[count].rsplit(' ', 1)[0])
        if os.path.isdir('./DTT_TO_CSV/target_missing'):
            pass
        else:
            os.mkdir('./DTT_TO_CSV/target_missing')
        new_filename = './DTT_TO_CSV/target_missing/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row

    time.sleep(2)

new_filename = './DTT_TO_CSV/questionnaire.csv'

df = pd.DataFrame(data_quest, columns=header_quest)
df.to_csv(new_filename, index=False)
