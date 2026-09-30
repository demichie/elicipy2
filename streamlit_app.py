import pandas as pd
import streamlit as st
import os.path
import sys

import bcrypt
from dotenv import load_dotenv

from github import Github
from github import Auth

from datetime import datetime

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

import secrets
import base64
import getpass

# from createWebformDict import *

import smtplib
from email import encoders
from email.mime.base import MIMEBase

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from io import StringIO


def send_email(sender, password, receiver, smtp_server, smtp_port,
               email_message, subject, attach_data, attach_name):

    message = MIMEMultipart()
    message['To'] = Header(receiver)
    message['From'] = Header(sender)
    message['Subject'] = Header(subject)
    message.attach(MIMEText(email_message, 'plain', 'utf-8'))

    # Add the attachment to the message
    f = StringIO()
    f.write(attach_data)
    f.seek(0)

    msg = MIMEBase('application', "octet-stream")
    msg.set_payload(f.read())
    encoders.encode_base64(msg)
    msg.add_header('Content-Disposition', 'attachment', filename=attach_name)
    message.attach(msg)

    server = smtplib.SMTP(smtp_server, smtp_port)
    print('server', server)
    server.starttls()
    server.ehlo()
    server.login(sender, password)
    text = message.as_string()
    server.sendmail(sender, receiver, text)
    server.quit()

    return


def generate_salt(size=16):
    """Generate the salt used for key derivation."""
    return secrets.token_bytes(size)


def derive_key(salt, password):
    """Derive the key from the password using the passed salt."""
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(password.encode())


def load_salt():
    return open("salt.salt", "rb").read()


def generate_key(password,
                 salt_size=16,
                 load_existing_salt=False,
                 save_salt=True):
    """Generate a key from a password and a salt."""
    if load_existing_salt:
        salt = load_salt()
    elif save_salt:
        salt = generate_salt(salt_size)
        with open("salt.salt", "wb") as salt_file:
            salt_file.write(salt)

    derived_key = derive_key(salt, password)
    return base64.urlsafe_b64encode(derived_key)


def encrypt(filename, key):
    """Encrypt a file in place."""
    f = Fernet(key)
    with open(filename, "rb") as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(filename, "wb") as file:
        file.write(encrypted_data)


def convert_df(df):
    return df.to_csv().encode('utf-8')


def pushToGithub(RepositoryData, df_new, csv_file, quest_type, datarepo):

    if datarepo == 'github':
        auth = Auth.Token(st.secrets["github_token"])

    elif datarepo == 'local_github':
        from createWebformDict import github_token
        auth = Auth.Token(github_token)

    g = Github(auth=auth)

    print('RepositoryData', RepositoryData)

    try:
        repo = g.get_user().get_repo(RepositoryData)
    except Exception:
        print('Repository not found: ', RepositoryData)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    git_prefix = './' + quest_type + '/'
    git_file = git_prefix + \
        csv_file.replace('.csv', '_') + dt_string + '_Output.csv'
    df2 = df_new.to_csv(sep=',', index=False)

    try:
        print('Committing file: ', git_file)
        repo.create_file(git_file, "committing files", df2, branch="main")
        st.write(git_file + ' CREATED')
        print(git_file + ' CREATED')
    except Exception:
        print('Problem committing file')

    return git_file


def saveAnswer(df_new, input_dir, csv_file, quest_type):

    output_dir = input_dir + '/' + quest_type
    isExist = os.path.exists(output_dir)

    if not isExist:
        os.makedirs(output_dir)
        print('The new directory ' + output_dir + ' is created!')

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    save_prefix = output_dir + '/'
    save_file = save_prefix + \
        csv_file.replace('.csv', '_') + dt_string + '_Output.csv'

    df_new.to_csv(save_file, sep=',', index=False)

    st.write(save_file + ' SAVED')

    return csv_file.replace('.csv', '_') + dt_string + '_Output.csv'


def check_form(qst, idxs, s, ans, units, minVals, maxVals, idx_list, idxMins,
               idxMaxs, sum50s, questions, label_flag, labels):

    print(ans[0:3])

    # IMPORTANT: qst and ans contain only the three percentile answers.
    # ADDITIONAL_INFO answers are deliberately stored in separate lists so
    # that this validation code keeps exactly the same indexing as before.
    n_qst = int((len(qst) - 2) / 3)

    check_flag = True

    for i in range(n_qst):

        if idxs[i] in idx_list:

            idx = 3 + i * 3

            if ',' in ans[idx]:

                if label_flag:
                    st.markdown('**Error in question ' + labels[i] + '**')
                else:
                    st.markdown('**Error in question ' + str(idxs[i]) + '**')

                st.write('Please remove comma')
                st.write(qst[idx], ans[idx])
                check_flag = False

            try:
                float(ans[idx])
            except ValueError:

                if label_flag:
                    st.markdown('**Error in question ' + labels[i] + '**')
                else:
                    st.markdown('**Error in question ' + str(idxs[i]) + '**')

                st.write('Non numeric answer')
                st.write(qst[idx], ans[idx])
                check_flag = False

            try:
                float(ans[idx + 1])
            except ValueError:

                if label_flag:
                    st.markdown('**Error in question ' + labels[i] + '**')
                else:
                    st.markdown('**Error in question ' + str(idxs[i]) + '**')

                st.write('Non numeric answer')
                st.write(qst[idx + 1], ans[idx + 1])
                check_flag = False

            try:
                float(ans[idx + 2])
            except ValueError:

                if label_flag:
                    st.markdown('**Error in question ' + labels[i] + '**')
                else:
                    st.markdown('**Error in question ' + str(idxs[i]) + '**')

                st.write('Non numeric answer')
                st.write(qst[idx + 2], ans[idx + 2])
                check_flag = False

            if check_flag:

                if float(ans[idx]) >= float(ans[idx + 1]):

                    if label_flag:
                        st.markdown('**Error in question ' + labels[i] + '**')
                    else:
                        st.markdown('**Error in question ' + str(idxs[i]) +
                                    '**')

                    st.write(qst[idx] + ' >= ' + qst[idx + 1])
                    check_flag = False

                if float(ans[idx + 1]) >= float(ans[idx + 2]):

                    if label_flag:
                        st.markdown('**Error in question ' + labels[i] + '**')
                    else:
                        st.markdown('**Error in question ' + str(idxs[i]) +
                                    '**')

                    st.write(qst[idx + 1] + ' >= ' + qst[idx + 2])
                    check_flag = False

                if float(ans[idx]) <= minVals[i] or float(
                        ans[idx]) >= maxVals[i]:

                    if label_flag:
                        st.markdown('**Error in question ' + labels[i] + '**')
                    else:
                        st.markdown('**Error in question ' + str(idxs[i]) +
                                    '**')

                    st.write(qst[idx] + ':' + str(ans[idx]))
                    st.write('The answer must be a value >' + str(minVals[i]) +
                             ' and  <' + str(maxVals[i]))
                    check_flag = False

                if float(ans[idx + 1]) <= minVals[i] or float(
                        ans[idx + 1]) >= maxVals[i]:

                    if label_flag:
                        st.markdown('**Error in question ' + labels[i] + '**')
                    else:
                        st.markdown('**Error in question ' + str(idxs[i]) +
                                    '**')

                    st.write(qst[idx + 1] + ':' + str(ans[idx + 1]))
                    st.write('The answer must be a value  >' +
                             str(minVals[i]) + ' and <' + str(maxVals[i]))
                    check_flag = False

                if float(ans[idx + 2]) <= minVals[i] or float(
                        ans[idx + 2]) >= maxVals[i]:

                    if label_flag:
                        st.markdown('**Error in question ' + labels[i] + '**')
                    else:
                        st.markdown('**Error in question ' + str(idxs[i]) +
                                    '**')

                    st.write(qst[idx + 2] + ':' + str(ans[idx + 2]))
                    st.write('The answer must be a value >' + str(minVals[i]) +
                             ' and <' + str(maxVals[i]))
                    check_flag = False

                if idxMins[i] < idxMaxs[i]:

                    sum50check = 0.0

                    for ii in range(idxMins[i] - 1, idxMaxs[i]):
                        sum50check += float(ans[4 + ii * 3])

                    if float(sum50s[i] != sum50check):

                        if label_flag:

                            labelIdxMin = idxs.index(idxMins[i])
                            labelIdxMax = idxs.index(idxMaxs[i])

                            st.markdown('**Error in question ' + labels[i] +
                                        '**')
                            st.write(
                                'Error in sum of 50%iles for questions from ',
                                labels[labelIdxMin], ' to ',
                                labels[labelIdxMax])

                        else:
                            st.markdown('**Error in question ' + str(idxs[i]) +
                                        '**')
                            st.write(
                                'Error in sum of 50%iles for questions from ',
                                idxMins[i], ' to ', idxMaxs[i])

                        st.write('The sum should be ' + str(sum50s[i]))
                        check_flag = False

    return check_flag


def additional_output_column(quest_type, idx, label, label_flag):
    """Return a stable, unique CSV column name for additional information."""
    if label_flag:
        return str(label) + '. Additional information'

    if quest_type == 'target':
        return 'TQ' + str(idx) + '. Additional information'

    return 'SQ' + str(idx) + '. Additional information'


def build_output_dataframe(qst, ans, additional_qst, additional_ans):
    """
    Build the individual-expert output DataFrame.

    The validation lists qst/ans contain only the three percentile values per
    question.  When ADDITIONAL_INFO is present in questionnaire.csv, one
    additional column per question is inserted immediately after its three
    percentile columns.
    """
    if len(additional_qst) == 0:
        return pd.DataFrame([ans], columns=qst)

    output_qst = qst[:3]
    output_ans = ans[:3]

    n_questions = int((len(qst) - 3) / 3)

    for i in range(n_questions):
        first = 3 + i * 3
        last = first + 3

        output_qst.extend(qst[first:last])
        output_ans.extend(ans[first:last])

        output_qst.append(additional_qst[i])
        output_ans.append(additional_ans[i])

    return pd.DataFrame([output_ans], columns=output_qst)


# Load environment variables from .env file
load_dotenv()
PASSWORD_HASH = os.getenv("PASSWORD_HASH")


def check_password():
    """Verify the entered password against the stored hash."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if bcrypt.checkpw(password.encode(), PASSWORD_HASH.encode()):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password! Please try again.")
        return False
    return True


def main():

    st.set_page_config(page_title="Elicipy", page_icon="logo.png")

    try:
        from createWebformDict import password_protected
    except ImportError:
        password_protected = False

    if password_protected:
        if check_password():
            show_form()
    else:
        show_form()


def show_form():

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Elicitation form")

    current_path = os.getcwd()

    path = current_path + '/ELICITATIONS'
    os.chdir(path)
    print('Path: ', path)
    sys.path.append(path)

    elicitation_list = next(os.walk(path))

    if len(elicitation_list[1]) == 1:
        wrk_dir = elicitation_list[1][0]
    else:
        try:
            from ElicitationCase import wrk_dir
        except ImportError:
            filename = path + '/ElicitationCase.py'
            isExist = os.path.exists(filename)

            if isExist:
                print('Please add wrk_dir to ElicitationCase.py')
            else:
                print('Please create file ElicitationCase.py with wrk_dir')

    path = current_path + '/ELICITATIONS/' + wrk_dir
    print('Path: ', path)
    sys.path.append(path)

    os.chdir(current_path)

    from createWebformDict import quest_type

    try:
        from createWebformDict import datarepo
    except ImportError:
        datarepo = 'local'

    print('Data repository:', datarepo)

    if (datarepo == 'local') or (datarepo == 'local_github'):
        input_dir = path + '/DATA'
        isExist = os.path.exists(input_dir)

        if not isExist:
            print('Please create Repository folder')
    else:
        input_dir = path + '/DATA'

    csv_file = 'questionnaire.csv'

    try:
        from createWebformDict import encrypted
    except ImportError:
        encrypted = False

    print('Encrypting of data:', encrypted)

    if encrypted:
        if (datarepo == 'local') or (datarepo == 'local_github'):
            password = getpass.getpass("Enter the password for encryption: ")
            key = generate_key(password, load_existing_salt=False)

    # Check if the PDF supporting file is defined and exists.
    try:
        from createWebformDict import companion_document
        pdf_doc = input_dir + '/' + companion_document
        isExists = os.path.exists(pdf_doc)
    except ImportError:
        isExists = False

    if isExists:
        with open(pdf_doc, "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Download PDF Questionnaire",
                           data=PDFbyte,
                           file_name=companion_document,
                           mime='application/octet-stream')

    # Check supplementary documents.
    try:
        from createWebformDict import supplementary_documents
        isExists = True
    except ImportError:
        isExists = False

    if isExists:
        for doc in supplementary_documents:
            pdf_doc = input_dir + '/' + doc
            isExists = os.path.exists(pdf_doc)

            if isExists:
                with open(pdf_doc, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()

                st.download_button(label="Download " + doc,
                                   data=PDFbyte,
                                   file_name=doc,
                                   mime='application/octet-stream')

    # Read the questionnaire to a pandas dataframe.
    df = pd.read_csv(input_dir + '/' + csv_file,
                     header=0,
                     index_col=0,
                     converters={'LABEL': lambda x: x.strip()})

    # ADDITIONAL_INFO is optional. Old questionnaire files without this column
    # retain exactly the previous form/output behaviour.
    has_additional_info = 'ADDITIONAL_INFO' in df.columns

    print('Questionnaire columns:', list(df.columns))
    print('ADDITIONAL_INFO available:', has_additional_info)

    if quest_type == 'seed':
        try:
            from createWebformDict import seed_list
            print('seed_list read', seed_list)
            idx_list = seed_list
        except ImportError:
            print('ImportError')
            idx_list = list(df.index)

    if quest_type == 'target':
        try:
            from createWebformDict import target_list
            print('seed_list read', target_list)
            idx_list = target_list
        except ImportError:
            print('ImportError')
            idx_list = list(df.index)

    if len(idx_list) == 0:
        idx_list = list(df.index)

    data_top = df.head()

    langs = []

    for head in data_top:
        if 'LONG Q' in head:
            string = head.replace('LONG Q', '')
            string2 = string.replace('_', '')
            langs.append(string2)

    print('langs', langs)

    if len(langs) > 1:
        options = langs
        lang_index = st.selectbox("Language",
                                  range(len(options)),
                                  format_func=lambda x: options[x])
        print('lang_index', lang_index)
        language = options[lang_index]
        # Select IDX, LABEL, SHORT Q, the chosen LONG Q language, and all
        # remaining questionnaire fields through ADDITIONAL_INFO.
        # With ADDITIONAL_INFO the structure contains 16 selected fields.
        if has_additional_info:
            index_list = [0, 1, 2, lang_index + 3] + \
                list(range(len(langs) + 3, len(langs) + 15))
        else:
            index_list = [0, 1, 2, lang_index + 3] + \
                list(range(len(langs) + 3, len(langs) + 14))
        print('language', language)
    else:
        lang_index = 0
        language = ''
        if has_additional_info:
            index_list = list(range(0, 16))
        else:
            index_list = list(range(0, 15))

    try:
        from createWebformDict import group_list
        print('group_list read', group_list)

        group = st.multiselect('Select your group', group_list, [])
        indices = [str(i + 1) for i, w in enumerate(group_list) if w in group]
        group = ';'.join(indices)

        print('Group', group, indices)
    except ImportError:
        print('ImportError group_list')
        group = '0'

    pctls = [5, 50, 95]

    form2 = st.form(key='form2', enter_to_submit=False)

    # qst/ans are deliberately reserved for personal data + percentiles only,
    # because check_form() assumes exactly three answer fields per question.
    ans = []

    qst = ["First Name"]
    ans.append(form2.text_input(qst[-1]))

    qst.append("Last Name")
    ans.append(form2.text_input(qst[-1]))

    qst.append("Email address")
    ans.append(form2.text_input(qst[-1]))

    # Additional-information columns/answers are kept separate from qst/ans.
    additional_qst = []
    additional_ans = []

    try:
        from createWebformDict import label_flag
    except ImportError:
        label_flag = False

    idxs = []
    labels = []
    units = []
    minVals = []
    maxVals = []

    idxMins = []
    idxMaxs = []
    sum50s = []

    questions = []

    for i in df.itertuples():

        if has_additional_info:
            idx, label, shortQ, longQ, unit, scale, minVal, maxVal, \
                realization, question, idxMin, idxMax, sum50, parent, image, \
                additional_info = [i[j] for j in index_list]
        else:
            idx, label, shortQ, longQ, unit, scale, minVal, maxVal, \
                realization, question, idxMin, idxMax, sum50, parent, image = \
                [i[j] for j in index_list]
            additional_info = None

        if question == quest_type:
            labels.append(str(label))
            idxs.append(idx)

    for i in df.itertuples():

        print([i[j] for j in index_list])

        if has_additional_info:
            idx, label, shortQ, longQ, unit, scale, minVal, maxVal, \
                realization, question, idxMin, idxMax, sum50, parent, image, \
                additional_info = [i[j] for j in index_list]
        else:
            idx, label, shortQ, longQ, unit, scale, minVal, maxVal, \
                realization, question, idxMin, idxMax, sum50, parent, image = \
                [i[j] for j in index_list]
            additional_info = None

        if question == quest_type:
            print('Question', idx, 'ADDITIONAL_INFO:', repr(additional_info))

        minVal = float(minVal)
        maxVal = float(maxVal)
        label = str(label)

        # Normalize UNITS. Empty CSV cells are read by pandas as NaN;
        # treat them as dimensionless quantities and display no unit suffix.
        if pd.isnull(unit):
            unit = ''
        else:
            unit = str(unit).strip()

        unit_text = ' [' + unit + ']' if unit else ''

        if question == quest_type:

            units.append(unit)

            if minVal.is_integer():
                minVal = int(minVal)

            if maxVal.is_integer():
                maxVal = int(maxVal)

            minVals.append(minVal)
            maxVals.append(maxVal)

            questions.append(questions)

            sum50 = float(sum50)

            idxMins.append(int(idxMin))
            idxMaxs.append(int(idxMax))
            sum50s.append(sum50)

            if idx in idx_list:

                form2.markdown("""___""")

                if quest_type == 'target':
                    if label_flag:
                        form2.header('TQ ' + label + '. ' + shortQ)
                    else:
                        form2.header('TQ' + str(idx) + '. ' + shortQ)
                else:
                    if label_flag:
                        form2.header('SQ ' + label + '. ' + shortQ)
                    else:
                        form2.header('SQ' + str(idx) + '. ' + shortQ)

                if not pd.isnull(image):
                    imagefile = input_dir + '/images/' + str(image)
                    if os.path.exists(imagefile):
                        form2.image(input_dir + '/images/' + str(image))

                if idxMin < idxMax:

                    if label_flag:
                        labelIdxMin = idxs.index(idxMin)
                        labelIdxMax = idxs.index(idxMax)

                        sum50_text = str(sum50) + (" " + unit if unit else "")
                        longQ_NB = "**N.B.** *The sum of 50%iles for " + \
                            "questions " + \
                            labels[labelIdxMin] + "-" + labels[labelIdxMax] + \
                            " have to sum to " + sum50_text + ".*"
                    else:
                        sum50_text = str(sum50) + (" " + unit if unit else "")
                        longQ_NB = "**N.B.** *The sum of 50%iles for " + \
                            "questions " + \
                            str(idxMin) + "-" + str(idxMax) + \
                            " have to sum to " + sum50_text + ".*"

                    form2.markdown(longQ)
                    form2.markdown(longQ_NB)
                else:
                    form2.markdown(longQ)

            for pct in pctls:

                qst.append(shortQ + ' - ' + str(int(pct)) + '%ile (' +
                           str(minVal) + ';' + str(maxVal) + ')' +
                           unit_text)

                if idx in idx_list:
                    ans.append(form2.text_input(qst[-1]))
                else:
                    ans.append('')

            # ------------------------------------------------------------
            # Optional free-form additional information field
            # ------------------------------------------------------------
            # If questionnaire.csv contains ADDITIONAL_INFO, the output CSV
            # contains one additional column for every question.  The box is
            # displayed only when the corresponding ADDITIONAL_INFO cell is
            # non-empty.  No validation is applied to this field.
            if has_additional_info:

                additional_qst.append(
                    additional_output_column(quest_type, idx, label,
                                             label_flag))

                show_additional_box = (not pd.isnull(additional_info)
                                       and str(additional_info).strip() != '')

                if idx in idx_list and show_additional_box:
                    additional_ans.append(
                        form2.text_area(str(additional_info),
                                        height=100,
                                        key='additional_info_' + quest_type +
                                        '_' + str(idx)))
                else:
                    additional_ans.append('')

    form2.markdown("""___""")

    agree_text = 'By sending this form and clicking the option “I AGREE”, ' \
                 + 'you hereby consent to the processing of your given ' \
                 + 'personal data (first name, last name and email address) ' \
                 + 'voluntarily provided. These data are used for the only ' \
                 + 'purpose of associating the asnwers of the seed question ' \
                 + 'to those of the target questions, and to communicate ' \
                 + 'with the participant only for matters related to the ' \
                 + 'expert elicitation. In accordance with the EU GDPR, ' \
                 + 'your personal data will be stored on a privite Github ' \
                 + 'repository (https://github.com/security) for as long ' \
                 + 'as is necessary for the purposes for which the personal ' \
                 + 'data are processed.'

    agree = form2.checkbox('I AGREE')

    form2.write(agree_text)

    form2.markdown("""___""")

    submit_button2 = form2.form_submit_button("Submit")

    if submit_button2:

        print('checkin for problems in answers')

        check_flag = check_form(qst, idxs, labels, ans, units, minVals,
                                maxVals, idx_list, idxMins, idxMaxs, sum50s,
                                questions, label_flag, labels)

        print('check_flag', check_flag)

        if not agree:
            st.write('Please agree to the terms above')

        if check_flag and agree:

            st.write('Thank you ' + ans[0] + ' ' + ans[1])

            from createWebformDict import confirmation_email

            if confirmation_email:

                if datarepo == 'github':
                    SENDER_ADDRESS = st.secrets["SENDER_ADDRESS"]
                    SENDER_PASSWORD = st.secrets["SENDER_PASSWORD"]
                    SENDER_NAME = st.secrets["SENDER_NAME"]
                    SMTP_SERVER_ADDRESS = st.secrets["SMTP_SERVER_ADDRESS"]
                    PORT = st.secrets["PORT"]
                else:
                    from createWebformDict import SENDER_ADDRESS
                    from createWebformDict import SENDER_PASSWORD
                    from createWebformDict import SENDER_NAME
                    from createWebformDict import SMTP_SERVER_ADDRESS
                    from createWebformDict import PORT

            if confirmation_email:
                st.write('Please check your email to see if you received a ' +
                         'confirmation message.')
                st.write(
                    "If you haven't received it, please wait a few minutes " +
                    "and submit your answers again.")

                email = ans[2]
                message = 'Dear ' + ans[0] + ' ' + ans[1] + \
                          ',\nThank you for filling in the questionnaire.\n' \
                          + 'You can find your answers attached to the ' + \
                          'email.\n' + 'Kind regards,\n' + SENDER_NAME
                subject = 'Elicitation confirmation'

            # Build the output only after validation. The additional fields
            # are inserted into the CSV here, so they never enter check_form().
            df_new = build_output_dataframe(qst, ans, additional_qst,
                                            additional_ans)
            df_new.insert(loc=3, column='Group(s)', value=group)

            if encrypted:
                f = Fernet(key)
                df_new = f.encrypt(df_new)

            if datarepo == 'github' or datarepo == 'local_github':

                from createWebformDict import RepositoryData

                print('Before pushing file to Gihub')
                save_file = pushToGithub(RepositoryData, df_new, csv_file,
                                         quest_type, datarepo)
                print('After pushing file to Gihub')
            else:
                print('Before saving file')
                save_file = saveAnswer(df_new, input_dir, csv_file, quest_type)
                print('After saving file')

            if confirmation_email:

                try:
                    send_email(sender=SENDER_ADDRESS,
                               password=SENDER_PASSWORD,
                               receiver=email,
                               smtp_server=SMTP_SERVER_ADDRESS,
                               smtp_port=PORT,
                               email_message=message,
                               subject=subject,
                               attach_data=df_new.to_csv(sep=',', index=False),
                               attach_name=save_file)
                except Exception:
                    print("Problem sending confirmation email")


if __name__ == '__main__':
    main()
