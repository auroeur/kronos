from torch.utils.data import Dataset
from kronos.lib.helpers import time_me 

class BinaryEncodedDataset(Dataset):
    def __init__(self, traces, attribute_mapper, attribs2use ):
        self.data = traces
        self.attrmap = attribute_mapper
        self.attribs2use = attribs2use

        # Maps attribute to required number of binary digits for encoding
        self.digits4attribute = {} 
        for a in self.attribs2use:
            self.digits4attribute[a] =  number_of_binary_digits_required(self.attrmap.get_attribute_size(a) + 1) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.to_binary_encoded(index)

    def input_size(self):
        return sum([count for _,count in self.digits4attribute.items()])

    def output_size(self):
        return self.attrmap.get_attribute_size('m_activity')

    def to_binary_encoded(self, index):
        val = self.data[index]
        
        # X must be a list of binary lists e.g. [[0,1], [1,0], [0,0]]
        X = []
        for e in val['event_sequence']:
            e_X = []
            for a in self.attribs2use:
                ctg_index = self.attrmap.get_index(a, e[a]) + 1 # +1 to avoid a zero vector for valid and known attribute value
                binary_rep = integer_to_binary_list(ctg_index, self.digits4attribute[a]) 
                e_X += binary_rep # join sub-vectors
            X.append(e_X)


        Y = self.attrmap.get_index('m_activity', val['next_activity'])
    
        return { 
            'trace': self.data[index],
            'X_raw': self.data[index]['event_sequence'], 
            'Y_raw': self.data[index]['next_activity'],
            'X': X,
            'Y': Y
            }

    def events_to_input(self, event_sequence):
        X = []
        for e in event_sequence:
            e_X = []
            for a in self.attribs2use:
                ctg_index = self.attrmap.get_index(a, e[a]) + 1 # +1 to avoid a zero vector for valid and known attribute value
                binary_rep = integer_to_binary_list(ctg_index, self.digits4attribute[a]) 
                e_X += binary_rep # join sub-vectors
            X.append(e_X)

        return X 

    def index_to_activity_name(self, index):
        return self.attrmap.get_activity_name(index)   


def number_of_binary_digits_required(number_of_classes):
    for power in range(16):
        if number_of_classes < 2**power:
            return power
    raise ValueError("Increase the range for the `power` candidates!")


def integer_to_binary_list(integer, number_of_digits):
    """ Convert an integer to a binary number as list of binary digits (zero and ones) """
    assert integer < 2**number_of_digits, "Integer must be less than 2 to the power of number_of_digits!"
    binary_list = [int(digit) for digit in "{:b}".format(integer)]
    leading_zeros = [0 for _ in range(number_of_digits - len(binary_list))] 
    full_binary_list = leading_zeros + binary_list
    assert len(full_binary_list) == number_of_digits
    return full_binary_list
