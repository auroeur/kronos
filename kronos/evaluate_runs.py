import os
import re
import pickle
import pandas as pd
import datetime 
from pprint import pprint as pp

import kronos.lib.metrics as m

FILE_DIR = (os.path.dirname(__file__))
BASEPATH = os.path.join(FILE_DIR,'..','simulation_results')

def extract_values(d):
    return [ 
        d['accuracy'],
        d['mcc'], 
        d['weighted avg']['f1-score'], 
        d['weighted avg']['precision'], 
        d['weighted avg']['recall'] 
    ]

COLUMN_NAMES = ['ACC', 
                'MCC', 
                'WA-F1', 
                'WA-PR', 
                'WA-RC']


def summary(files, initial_split_time):
    # key values as `row_name:[]`
    # list of lists 
    rows_data = []

    for fn in files:
        data = pickle.load(open(BASEPATH+fn,'rb'))

        _traces = data['traces']
        model_history = data['model_history']

        for t in _traces:
            if t["last_event_timestamp"] <= initial_split_time:
                assert t["prediction_model_id"] == None
                assert t["predicted_next_activity"] == None

        # Limit to only future traces 
        traces = [ t for t in _traces if t["last_event_timestamp"] > initial_split_time ]
        
        number_of_unpredicted_nact = len([ t for t in traces if t['predicted_next_activity'] == None ])
        number_of_predicted_nact = len(traces) - number_of_unpredicted_nact

        y_true = []
        y_pred = []
        y_model = []
        y_index = []

        unpredicted = []

        index = 0
        first_non_not = False
        for t in traces:
            if t['prediction_model_id'] != None:
                index += 1
                y_index.append(index)
                y_pred.append(t['predicted_next_activity'])
                y_true.append(t['next_activity'])
                y_model.append(t['prediction_model_id'])
            else:
                unpredicted.append(t)
        
        # Detailed
        # #####################################################
        _scores = m.classification_scores(y_true=y_true, y_pred=y_pred)
        class_report = _scores['classification_report']
        class_report['mcc'] = _scores['mcc']
        class_report['balanced_accuracy'] = _scores['balanced_accuracy']

        no_models = len(model_history)
        total_training_time = round( sum([i['training_time'] for i in model_history.values()]) )
        total_training_time_str = str(datetime.timedelta(seconds=total_training_time))

        dataset_plus_alog_label = fn.split('_')[0]+'-'+re.search('simulation_0.',fn)[0][-1]
        rows_data.append([ dataset_plus_alog_label, *extract_values(class_report), number_of_unpredicted_nact, number_of_predicted_nact, no_models, total_training_time_str])


    
    df = pd.DataFrame(rows_data, columns=['datasetSim', *COLUMN_NAMES, '\#NP', '\#P', '\#Model', 'Time']).round(3)
    df_str = df.astype(str)

    for col in df:
        if col in ['datasetSim', '\#NP', '\#P', '\#Model', 'Time']:
            continue
        max_val_idx  = df[col].idxmax()
        df_str[col][max_val_idx] = f"\textbf{{{df[col][max_val_idx]}}}"  

    return df_str


def print_prediction_metrics(func):
    print(pd.concat([
        func([
        '/helpdesk_simulation_00.pickle',
        '/helpdesk_simulation_01.pickle',
        '/helpdesk_simulation_02.pickle',
        '/helpdesk_simulation_03.pickle',
        ], initial_split_time="2010-07-29 23:59:59"),

        func([
        '/bpi2012_simulation_00.pickle',
        '/bpi2012_simulation_01.pickle',
        '/bpi2012_simulation_02.pickle',
        '/bpi2012_simulation_03.pickle',
        ], initial_split_time="2011-10-21 23:59:59"),

        func([
        '/sepsis_simulation_00.pickle',
        '/sepsis_simulation_01.pickle',
        '/sepsis_simulation_02.pickle',
        '/sepsis_simulation_03.pickle',
        ], initial_split_time="2014-01-17 23:59:59"),
    ]).to_latex(escape=False, index=False))



if '__main__'==__name__:
    print_prediction_metrics(summary)
        

