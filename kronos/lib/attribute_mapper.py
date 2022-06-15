from kronos.lib.dataset_encoders import *
from collections import defaultdict 


def HelpdeskAttributeMapper(traces):
    categorical_attributes = [
        'm_activity', 
        'customer', 
        'product', 
        'resource', 
        'responsible_section', 
        'seriousness', 
        'seriousness_2', 
        'service_level', 
        'service_type', 
        'variant', 
        'workgroup', 
        'support_section' 
    ] 
    continuous_attributes = [] 

    return Attribute2IndexMapper(traces, categorical_attributes, continuous_attributes) 

def BPI2012AttributeMapper(traces):
    categorical_attributes = [
        'm_activity', 
        'org_resource'
    ] 
    continuous_attributes = [
        'trace_amount_req'  
    ]
    return Attribute2IndexMapper(traces, categorical_attributes, continuous_attributes) 
 

def SepsisAttributeMapper(traces):
    categorical_attributes = [
        'm_activity', 
        'Hypoxie',
        'InfectionSuspected',
        'DiagnosticLiquor',
        'SIRSCritHeartRate',
        'DisfuncOrg',
        'Hypotensie',
        'org:group',
        'DiagnosticOther',
        'DiagnosticIC',
        'DiagnosticXthorax',
        'Oligurie',
        'DiagnosticSputum',
        'SIRSCritTachypnea',
        'DiagnosticBlood',
        'SIRSCritTemperature',
        'Infusion',
        'SIRSCritLeucos',
        'Diagnose',
        'DiagnosticECG',
        'DiagnosticArtAstrup',
        'DiagnosticUrinaryCulture',
        'DiagnosticLacticAcid',
        'SIRSCriteria2OrMore',
        'DiagnosticUrinarySediment'
    ] 
    continuous_attributes = [
        'Age',  
        'Leucocytes',
        'LacticAcid',
        'CRP'
    ]
    return Attribute2IndexMapper(traces, categorical_attributes, continuous_attributes) 
 



class Attribute2IndexMapper:
    """ Maps attribute values to 0-based indices. Unknown values are mapped to index -1. """
    def __init__(self, traces, categorical_attributes, continuous_attributes):

        self.categorical_attributes = categorical_attributes
        self.continuous_attributes = continuous_attributes
       
        self.ctg_attribute_map = {
            **create_maps_for_categorical_attributes(traces, self.categorical_attributes),
            **create_maps_for_continuous_attributes(traces, self.continuous_attributes),
            'm_delta_seconds': TimeDelta2Index()
        }

        self.size_for_attributes  = { k:v.number_of_categories() for k,v in self.ctg_attribute_map.items() }
            
    def get_index(self, attribute, value):
        if attribute in self.ctg_attribute_map:
            return self.ctg_attribute_map[attribute].get_category_index(value) 
        else:
            raise ValueError(f"Attribute {attribute} doesn't exist")

    def get_attribute_size(self, attribute):
        return self.size_for_attributes[attribute]

    def get_activity_name(self, index):
        return self.ctg_attribute_map['m_activity'].get_category_by_index(index)

    def get_all_activites(self):
        return [ act for act in self.ctg_attribute_map['m_activity'].categories ]



def create_maps_for_categorical_attributes(traces, categorical_attributes):
    """ Creates map of maps: attributes -> ( categories -> index ) """
    
    # maps attribute -> set(categories)
    dict_attr = { c:set() for c in categorical_attributes } 

    for t in traces:
        dict_attr['m_activity'].add(t['next_activity'])
        for e in t['event_sequence']:
            for attr in categorical_attributes:
                if e[attr]: # won't include attribute if 'None' or ''
                    dict_attr[attr].add(e[attr])

    dict_attr['m_activity'].add('END')
    
    # maps attribute -> ( categoriy -> index)
    ctg_attribute_map = {}
    for attr, categories in dict_attr.items():
        ctg_attribute_map[attr] = CategoricalValue2Index(categories)

    return ctg_attribute_map
 

def create_maps_for_continuous_attributes(traces, continuous_attributes):
    """ Creates map of maps: attributes -> ( categories -> index ) """
    
    # maps attribute -> list(values)
    attr_values = { c:[] for c in continuous_attributes }

    for t in traces:
        for e in t['event_sequence']:
            for attr in continuous_attributes:
                if e[attr]:
                    attr_values[attr].append(float(e[attr]))

    ctg_attribute_map = {}
    for attr, values in attr_values.items():
        ctg_attribute_map[attr] = ContinuousValue2Index(min_v=min(values), max_v=max(values), bins=2**4)
     
    return ctg_attribute_map
 

class TimeDelta2Index():
    def __init__(self):
        hour = 60*60
        day = hour*24
        week = day*7
        month = int(week*4.33)
        self.categories = [ hour, hour*4, hour*8, day, day*2, day*3, day*4, day*5, day*6,
                week, week*2, week*3, month, month*2, month*3 ]

    def get_category_index(self, seconds):
        for idx, category in enumerate(self.categories):
            if seconds <= category:
                return idx
        return len(self.categories) 

    def number_of_categories(self):
        return len(self.categories)


class ContinuousValue2Index():
    """ Maps numerical values to an index value.

    Splits range into n bins. A value is assgined to a bin where the value
    is <= than the bin's rhs border. We use #bins-1 borders with 
    `min_v + bin_size * i` with `i` in [1,...,#bins-1].
    The `bin_size` is determined by (max_v-min_v)/bins.
    
    Values greater than the last border are assgined to the last bin.

    For non numercial values we return -1.
    """
    def __init__(self, min_v, max_v, bins):
        assert bins >= 2 
        diff = max_v - min_v
        bin_size = diff/int(bins)
        self.categories = [ min_v+bin_size*b for b in range(1, bins) ]
        #print('min max', min_v, max_v, '#bins', bins)
        #print(self.categories, len(self.categories)+1)

    def get_category_index(self, numeric_value):
        if not numeric_value:
            return -1
        value = float(numeric_value)
        for idx, category in enumerate(self.categories):
            if value <= category:
                return idx
        return len(self.categories)

    def number_of_categories(self):
        return len(self.categories)



class CategoricalValue2Index():
    """ Maps categorical value to an index value.
    For unknown categories we return -1.
    """
    def __init__(self, categories):
        self.categories ={ val:idx for idx, val in enumerate(categories) } 
        self.indices = { idx:val for val, idx in self.categories.items() }

    def get_category_index(self, value):
        return self.categories.get(value, -1) 

    def number_of_categories(self):
        return len(self.categories)

    def get_category_by_index(self, index):
        return self.indices[index]
   
