import numpy as np
import pandas as pd


def MFH(guide_seq, off_seq, dim=5):
    code_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
                 '-': [0, 0, 0, 0, 1]}
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '-': 1}
    tlen = 24
    guide_seq = "-" * (tlen - len(guide_seq)) + guide_seq.upper()
    off_seq = "-" * (tlen - len(off_seq)) + off_seq.upper()

    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    on_encoded_matrix = np.zeros((24, 5), dtype=np.float32)
    off_encoded_matrix = np.zeros((24, 5), dtype=np.float32)

    on_off_dim7_codes = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]

        if gRNA_list[i] == '_':
            gRNA_list[i] = '-'

        if off_list[i] == '_':
            off_list[i] = '-'

        gRNA_base_code = code_dict[gRNA_list[i].upper()]
        DNA_based_code = code_dict[off_list[i].upper()]
        diff_code = np.bitwise_or(gRNA_base_code, DNA_based_code)

        if(dim==7):
            dir_code = np.zeros(2)
            if gRNA_list[i] == "-" or off_list[i] == "-" or direction_dict[gRNA_list[i]] == direction_dict[off_list[i]]:
                pass
            else:
                if direction_dict[gRNA_list[i]] > direction_dict[off_list[i]]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1

            pair_code.append(np.concatenate((diff_code, dir_code)))
        else:
            pair_code.append(diff_code)
        on_encoded_matrix[i] = code_dict[gRNA_list[i]]
        off_encoded_matrix[i] = code_dict[off_list[i]]

    pair_code_matrix = np.array(pair_code, dtype=np.float32).reshape(1, 1, 24, dim)

    return pair_code_matrix, on_encoded_matrix.reshape(1, 1, 24, 5), off_encoded_matrix.reshape(1, 1, 24, 5)




#  crispr-ip 7*24
def crispr_ip_coding(target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0],
                    '-': [0, 0, 0, 0]}
    pos_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, '_': 5, '-': 5}
    tlen = 24
    target_seq = "-" * (tlen - len(target_seq)) + target_seq.upper()
    off_target_seq = "-" * (tlen - len(off_target_seq)) + off_target_seq.upper()

    target_seq = list(target_seq)
    off_target_se = list(off_target_seq)

    for i in range(len(target_seq)):
        if target_seq[i] == 'N':
            target_seq[i] = off_target_seq[i]


    target_seq_code = np.array([encoded_dict[base] for base in target_seq])
    off_target_seq_code = np.array([encoded_dict[base] for base in off_target_se])
    on_off_dim6_codes = []
    for i in range(len(target_seq)):
        diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code * -1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24, 1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code

class Encoder():
    def __init__(self, on_seq, off_seq):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq.upper()
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq.upper()
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)

        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            if off_b == "N":
                off_b = on_b
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)

class New_Encoder():
    def __init__(self, on_seq, off_seq):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq.upper()
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq.upper()
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        self.encode_both()

    def encode_on(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)

        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.on_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_both(self):
        self.encode_on()
        self.encode_off()
        diff_code = np.bitwise_or(self.on_code, self.off_code)
        self.on_off_code = diff_code


#24*7
def crispr_net_coding(on_seq,off_seq):
    e = Encoder(on_seq=on_seq, off_seq=off_seq)
    return e.on_off_code


def New_coding(on_seq,off_seq):
    e = New_Encoder(on_seq=on_seq, off_seq=off_seq)
    return e.on_code,e.off_code,e.on_off_code

# 23*4
def cnn_predict(guide_seq, off_seq):

    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]

    if len(off_seq) == 24:
        off_seq = off_seq[1:]


    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]

        gRNA_list[i] = gRNA_list[i].upper()
        off_list[i] = off_list[i].upper()

        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
    input_code = np.array(pair_code).reshape(1, 1, 23, 4)
    return input_code

def cnn_oneseq(guide_seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    encoded_matrix = np.zeros((24, 4), dtype=int)
    for i, nucleotide in enumerate(guide_seq):
        encoded_matrix[i] = code_dict[nucleotide]
    return encoded_matrix.reshape(1, 24, 4)


#23*14
def dnt_coding(on_seq, off_seq):
    on_seq = on_seq.upper()
    off_seq = off_seq.upper()

    on_seq = list(on_seq)
    off_seq = list(off_seq)

    for i in range(len(off_seq)):
        if on_seq[i] == 'N':
            on_seq[i] = off_seq[i]

        if off_seq[i] == 'N':
            off_seq[i] = on_seq[i]

    on_seq = ''.join(on_seq)
    off_seq = ''.join(off_seq)

    # One-hot编码函数
    def one_hot_encode_seq(data):
        if len(data)==24:
            data = data[1:]
        alphabet = 'AGCT'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in data]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    arr1 = one_hot_encode_seq(on_seq)
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(off_seq)
    arr2 = np.asarray(arr2).T
    combined = np.concatenate((arr1, arr2))


    # 加区分碱基错配类型和区域划分
    encoded_list = np.zeros((5, 23))
    for m in range(23):
        arr1 = combined[0:4, m].tolist()
        arr2 = combined[4:8, m].tolist()
        arr = []
        if arr1 == arr2:
            arr = [0, 0, 0, 0, 0]
        else:
            arr = np.add(arr1, arr2).tolist()
            arr.append(1 if (arr == [1, 1, 0, 0] or arr == [0, 0, 1, 1]) else -1)
        encoded_list[:, m] = arr

    # 加区域分开编码
    position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    final_encoding = np.zeros((14, 23))
    for k in range(8):
        final_encoding[k] = combined[k]
    final_encoding[8:13] = encoded_list[0:5]
    final_encoding[13] = position

    final_encoding = final_encoding.reshape(14, 23).transpose((1,0))
    return final_encoding


def dnt_coding_24_14(on_seq, off_seq):

    on_seq = on_seq.upper()
    off_seq = off_seq.upper()

    on_seq = list(on_seq)
    off_seq = list(off_seq)

    for i in range(len(off_seq)):
        if on_seq[i] == 'N':
            on_seq[i] = off_seq[i]

        if off_seq[i] == 'N':
            off_seq[i] = on_seq[i]

        if on_seq[i] == '-':
            on_seq[i] = '_'

        if off_seq[i] == '-':
            off_seq[i] = '_'
    on_seq = ''.join(on_seq)
    off_seq = ''.join(off_seq)

    # One-hot编码函数
    def one_hot_encode_seq(data):
        if (len(data) == 23):
            data = '_'+data
        # if len(data)==24:
        #     data = data[1:]
        alphabet = 'AGCT_'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in data]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    arr1 = one_hot_encode_seq(on_seq)
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(off_seq)
    arr2 = np.asarray(arr2).T
    combined = np.concatenate((arr1, arr2))


    # 加区分碱基错配类型和区域划分
    encoded_list = np.zeros((5, 24))
    for m in range(24):
        arr1 = combined[0:4, m].tolist()
        arr2 = combined[4:8, m].tolist()
        arr = []
        if arr1 == arr2:
            arr = [0, 0, 0, 0, 0]
        else:
            arr = np.add(arr1, arr2).tolist()
            arr.append(1 if (arr == [1, 1, 0, 0] or arr == [0, 0, 1, 1]) else -1)
        encoded_list[:, m] = arr

    # 加区域分开编码
    position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    final_encoding = np.zeros((14, 24))
    for k in range(8):
        final_encoding[k] = combined[k]
    final_encoding[8:13] = encoded_list[0:5]
    final_encoding[13] = position

    final_encoding = final_encoding.reshape(14, 24).transpose((1,0))
    return final_encoding



encoding_map = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15,
    '-A': 16, '-C': 17,'-G': 18, '-T': 19,
    'A-': 20, 'C-': 21,'G-': 22, 'T-': 23,
    '--': 24
}
def word_Encoding(sgRNA, DNA):

    if len(sgRNA) == 24:
        sgRNA = sgRNA[1:]

    if len(DNA) == 24:
        DNA = DNA[1:]

    sgRNA = list(sgRNA)

    sgRNA[-3] = DNA[-3]
    DNA = list(DNA)
    for j in range(len(sgRNA)):
        if sgRNA[j] == 'N':
            sgRNA[j] = DNA[j]
        if DNA[j] == 'N':
            DNA[j] = sgRNA[j]

    pairs = [(sgRNA[i].upper() if sgRNA[i] != '_' else '-') + (DNA[i].upper() if DNA[i] != '_' else '-') for i in range(len(sgRNA))]

    return [encoding_map[p] for p in pairs]


def dl_offtarget(guide_seq, off_seq):
    """
    Modifies the encoding function to concatenate the encoded guide_seq and off_seq
    into a 23x8 matrix.
    """

    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]

    if len(off_seq) == 24:
        off_seq = off_seq[1:]

    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    gRNA_encoded = []
    off_encoded = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_list[i] = gRNA_list[i].upper()
        off_list[i] = off_list[i].upper()

        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        gRNA_encoded.append(gRNA_base_code)
        off_encoded.append(DNA_based_code)

    # Concatenate the encoded guide sequence and off-target sequence
    concatenated_code = np.concatenate((gRNA_encoded, off_encoded), axis=1)
    input_code = concatenated_code.reshape(1, 1, 23, 8)
    return input_code


def dl_crispr(guide_seq, off_seq):
    """
    Extends the encoding function to concatenate the encoded guide_seq and off_seq into a 23x8 matrix,
    and then further concatenates a 23x12 matrix to represent mismatch types between guide RNA and DNA sequences.
    """

    # Adjust sequences if they start with a length of 24
    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]
    if len(off_seq) == 24:
        off_seq = off_seq[1:]

    guide_seq = guide_seq[:-3]
    off_seq = off_seq[:-3]

    # Encoding dictionary for A, T, G, C
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

    # Mismatch encoding dictionary
    mismatch_dict = {
        'AC': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'AG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'AT': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'CA': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'CG': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'CT': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'GA': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'GC': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'GT': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    gRNA_list = list(guide_seq.upper())
    off_list = list(off_seq.upper())
    gRNA_encoded = []
    off_encoded = []
    mismatch_encoded = []

    for i in range(len(gRNA_list)):
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        gRNA_encoded.append(gRNA_base_code)
        off_encoded.append(DNA_based_code)

        # Encode mismatches
        mismatch_type = gRNA_list[i] + off_list[i]
        mismatch_code = mismatch_dict.get(mismatch_type, [0] * 12)  # Default to no mismatch
        mismatch_encoded.append(mismatch_code)

    # Concatenate the encoded guide sequence and off-target sequence
    concatenated_code = np.concatenate((gRNA_encoded, off_encoded), axis=1)

    # Further concatenate the mismatch encoding
    full_concatenated_code = np.concatenate((concatenated_code, mismatch_encoded), axis=1)
    input_code = full_concatenated_code.reshape(1, 1, 20, -1)  # Reshape according to your model's input requirement
    return input_code


from keras_bert import Tokenizer

class BertEncoder:
    def __init__(self, data, dim):
        self.data = data
        self.dim = dim
        self.encoded_dict_indel = {'aa': [1, 0, 0, 0, 0, 0, 0], 'at': [1, 1, 0, 0, 0, 1, 0],
                                   'ag': [1, 0, 1, 0, 0, 1, 0], 'ac': [1, 0, 0, 1, 0, 1, 0],
                                   'ta': [1, 1, 0, 0, 0, 0, 1], 'tt': [0, 1, 0, 0, 0, 0, 0],
                                   'tg': [0, 1, 1, 0, 0, 1, 0], 'tc': [0, 1, 0, 1, 0, 1, 0],
                                   'ga': [1, 0, 1, 0, 0, 0, 1], 'gt': [0, 1, 1, 0, 0, 0, 1],
                                   'gg': [0, 0, 1, 0, 0, 0, 0], 'gc': [0, 0, 1, 1, 0, 1, 0],
                                   'ca': [1, 0, 0, 1, 0, 0, 1], 'ct': [0, 1, 0, 1, 0, 0, 1],
                                   'cg': [0, 0, 1, 1, 0, 0, 1], 'cc': [0, 0, 0, 1, 0, 0, 0],
                                   'ax': [1, 0, 0, 0, 1, 1, 0], 'tx': [0, 1, 0, 0, 1, 1, 0],
                                   'gx': [0, 0, 1, 0, 1, 1, 0], 'cx': [0, 0, 0, 1, 1, 1, 0],
                                   'xa': [1, 0, 0, 0, 1, 0, 1], 'xt': [0, 1, 0, 0, 1, 0, 1],
                                   'xg': [0, 0, 1, 0, 1, 0, 1], 'xc': [0, 0, 0, 1, 1, 0, 1],
                                   'xx': [0, 0, 0, 0, 0, 0, 0]}
        self.encode()

    def encode(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        data_bases = list(self.data)
        j = 0
        code_list.append(encoded_dict['xx'])
        for i in range(self.dim):

            code_list.append(encoded_dict[data_bases[j] + data_bases[j + 1]])
            j = j + 3
        code_list.append(encoded_dict['xx'])
        self.on_off_code = np.array(code_list)


token_dict = {
    '[CLS]': 0,
    '[SEP]': 1, 'aa': 2, 'ac': 3, 'ag': 4, 'at': 5,
    'ca': 6, 'cc': 7, 'cg': 8, 'ct': 9,
    'ga': 10, 'gc': 11, 'gg': 12, 'gt': 13,
    'ta': 14, 'tc': 15, 'tg': 16, 'tt': 17,
    'ax': 18, 'xa': 19, 'cx': 20, 'xc': 21, 'gx': 22,
    'xg': 23, 'tx': 24, 'xt': 25, 'xx': 26
}
tokenizer = Tokenizer(token_dict)


def BERT_encode(data):

    idxs = list(range(len(data)))
    X1, X2 = [], []
    for i in idxs:
        text, y = data[i]

        # print(f"{i}:BERT_encode:", text)
        x1, x2 = tokenizer.encode(text)
        X1.append(x1)
        X2.append(x2)
    return X1, X2


def C_RNN_encode(data, dim):
    print("C_RNN_encode:", data)
    encode = []
    for idx, row in data.iterrows():
        # print(f"{idx}:C_RNN_encode:", row[0])
        en = BertEncoder(row[0], dim)
        encode.append(en.on_off_code)
    return encode

def encode_by_base_pair_vocabulary(on_target_seq, off_target_seq):
    BASE_PAIR_VOCABULARY_v1 = {
        "AA":0,  "TT":1,  "GG":2,  "CC":3,
        "AT":4,  "AG":5,  "AC":6,  "TG":7,  "TC":8,  "GC":9,
        "TA":10, "GA":11, "CA":12, "GT":13, "CT":14, "CG":15,
        "A_":16, "T_":17, "G_":18, "C_":19,
        "_A":20, "_T":21, "_G":22, "_C":23,
        "AAP":24,  "TTP":25,  "GGP":26,  "CCP":27,
        "ATP":28,  "AGP":29,  "ACP":30,  "TGP":31,  "TCP":32,  "GCP":33,
        "TAP":34, "GAP":35, "CAP":36, "GTP":37, "CTP":38, "CGP":39,
        "A_P":40, "T_P":41, "G_P":42, "C_P":43,
        "_AP":44, "_TP":45, "_GP":46, "_CP":47,
        "__":48, "__P":49
    }
    BASE_PAIR_VOCABULARY_v2 = {
        "AA": 0,    "TT": 1,    "GG": 2,    "CC": 3,
        "AAP":4,    "TTP":5,    "GGP":6,    "CCP":7,
        "AT": 8,    "AG": 9,    "AC": 10,   "TG": 11,   "TC": 12,   "GC": 13,
        "TA": 14,   "GA": 15,   "CA": 16,   "GT": 17,   "CT": 18,   "CG": 19,
        "ATP":20,   "AGP":21,   "ACP":22,   "TGP":23,   "TCP":24,   "GCP":25,
        "TAP":26,   "GAP":27,   "CAP":28,   "GTP":29,   "CTP":30,   "CGP":31,
        "A_": 32,   "T_": 33,   "G_": 34,   "C_": 35,
        "_A": 36,   "_T": 37,   "_G": 38,   "_C": 39,
        "A_P":40,   "T_P":41,   "G_P":42,   "C_P":43,
        "_AP":44,   "_TP":45,   "_GP":46,   "_CP":47,
        "__": 48,   "__P":49
    }
    BASE_PAIR_VOCABULARY_v3 = {
        "AA": 0,    "TT": 1,    "GG": 2,    "CC": 3,
        "AT": 4,    "AG": 5,    "AC": 6,   "TG": 7,   "TC": 8,   "GC": 9,
        "TA": 10,   "GA": 11,   "CA": 12,   "GT": 13,   "CT": 14,   "CG": 15,
        "A_": 16,   "T_": 17,   "G_": 18,   "C_": 19,
        "_A": 20,   "_T": 21,   "_G": 22,   "_C": 23,
        "__": 24
    }
    tlen = 24
 
    on_target_seq = "_"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "_"*(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq = on_target_seq.replace("-", "_")
    off_target_seq = off_target_seq.replace("-", "_")

    pair_vector = list()
    for i in range(tlen):
        base_pair = on_target_seq[i]+off_target_seq[i]
        # if i > 20:
        #     base_pair += "P"
        pair_vector.append(BASE_PAIR_VOCABULARY_v3[base_pair.upper()])
    pair_vector = np.array(pair_vector)
    return pair_vector

def encode_by_base_vocabulary(seq):
    BASE_VOCABULARY_v1 = {
        "A": 50, "T": 51, "G": 52, "C": 53, "_": 54
    }
    BASE_VOCABULARY_v3 = {
        "A": 25, "T": 26, "G": 27, "C": 28, "_": 29
    }
    tlen = 24
    
    seq = "_"*(tlen-len(seq)) + seq
    seq = seq.replace("-", "_")

    seq_vector = list()
    for i in range(tlen):
        base = seq[i].upper()
        seq_vector.append(BASE_VOCABULARY_v3[base])
    seq_vector = np.array(seq_vector)
    return seq_vector


def CRISPR_M(on_target_seq, off_target_seq):

    # on_target_seq = on_target_seq.upper()
    # off_target_seq = off_target_seq.upper

    gRNA_list = list(on_target_seq)

    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_target_seq[i]

    on_target_seq = ''.join(gRNA_list)

    pair_vector = encode_by_base_pair_vocabulary(on_target_seq, off_target_seq)
    off_vector = encode_by_base_vocabulary(on_target_seq)
    on_vector = encode_by_base_vocabulary(off_target_seq)
    return pair_vector, off_vector, on_vector



def deepCrispr(on_target_seq, off_target_seq):
    gRNA_list = list(on_target_seq)

    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_target_seq[i]

    on_target_seq = ''.join(gRNA_list)

    ntmap = {'A': (1.0, 0.0, 0.0, 0.0),
            'C': (0.0, 1.0, 0.0, 0.0),
            'G': (0.0, 0.0, 1.0, 0.0),
            'T': (0.0, 0.0, 0.0, 1.0)
            }

    tlen = 23
    on_target_seq_code = np.array([ntmap[base.upper()] for base in list(on_target_seq)])
    off_target_seq_code = np.array([ntmap[base.upper()] for base in list(off_target_seq)])

    return on_target_seq_code, off_target_seq_code
