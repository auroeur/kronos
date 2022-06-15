import os
import csv
import pickle 
from collections import defaultdict
from pprint import pprint as pp
from toolz.itertoolz import groupby
from kronos.lib.helpers import time_me, make_datetime, delta_seconds

import pm4py

FILE_DIR = (os.path.dirname(__file__))
DATA_OUTPUT_DIR = os.path.join(FILE_DIR,'..','data') # where transformed data will be dumped
DATA_RAW_DIR = os.path.join(DATA_OUTPUT_DIR,'raw') # where unmodified data is expected

LEN_TIMESTAMP = len('0000-00-00 00:00:00')

class XESData:
    def __init__(self, xes_filepath, pickle_filepath, use_cache=False):
        self.xes_file = xes_filepath
        self.pickle_file = pickle_filepath # see _pickle

        self.traces = None # list of traces

        if (not use_cache) or (not  os.path.exists(self.pickle_file)):
            print("Loading XES")
            self.traces = self._load_xes_from_xes_file()
            self._pickle()
        else:
            print("Loading XES from cache")
            self.traces = self._load_xes_from_pickle()
        
    @time_me
    def _pickle(self):
        # Pickling for faster access and loading
        pickle.dump(self.traces, open(self.pickle_file, 'wb'))

    @time_me
    def _load_xes_from_pickle(self):
        return pickle.load(open(self.pickle_file, 'rb'))

    @time_me
    def _load_xes_from_xes_file(self):
        return pm4py.read_xes(self.xes_file)


def drop_consecutive_repeating_trace_events(trace):
    """ Expects a list of events part of a trace. """
    if len(trace) == 0:
        return []
    if len( set([e['m_activity'] for e in trace]) ) == 1:
        return []

    keep_events = []

    print("===============")
    for e in trace:
        if len(keep_events) == 0:
            keep_events.append(e)
            print("Keeping ", e['m_case_id'], e['m_activity'])
        elif keep_events[-1]['m_activity'] == e['m_activity']:
            print("Dropping", e['m_case_id'], e['m_activity'])
            continue
        else:
            print("Keeping ", e['m_case_id'], e['m_activity'])
            keep_events.append(e)
    
    return keep_events

def mark_start_end_for_sorted_trace(trace):
    """ Expects a list of events part of a trace. """
    assert len(trace) > 1
    for e in trace:
        e['m_start_end'] = None
    trace[0]['m_start_end'] = 'start'
    trace[-1]['m_start_end'] = 'end'
     
def add_delta_seconds_since_first_event_to_trace_events(trace):
    """ Compute delta seconds since first event """
    init_timestamp = make_datetime(trace[0]['m_timestamp'])
    for e in trace:
        e['m_delta_seconds'] = delta_seconds(init_timestamp, make_datetime(e['m_timestamp']))



class EventLogLoader():
    def __init__(self, event_logs):
        self.event_logs = event_logs

    def pull_data(self, from_date=None, till_date=None):
        event_logs = self.event_logs
        assert event_logs != None
        if not from_date and not till_date:
            return event_logs
        elif from_date and till_date:
            return [ e for e in event_logs if e['m_timestamp'] >= from_date and e['m_timestamp'] <= till_date ]
        elif from_date and not till_date:
            return [ e for e in event_logs if e['m_timestamp'] >= from_date ]
        elif not from_date and till_date:
            return [ e for e in event_logs if e['m_timestamp'] <= till_date ]

    def min_max_date(self):
        """ Return min and maximum date from a sorted input of event logs """
        event_logs = self.event_logs
        return (event_logs[0]['m_timestamp'], event_logs[-1]['m_timestamp'])



def load_BPI2012(use_cache=False):
    
    xes = XESData(DATA_RAW_DIR+'/bpi2012/financial_log.xes', DATA_OUTPUT_DIR+'/tmp.bpi2012.pickle',use_cache=use_cache)
    traces = xes.traces

    final_events = []

    for idx, trace in enumerate(traces):
        # We sort using the full fractional seconds available but bellow
        # cut the fractional seconds because python's sorting is stable. 
        events = sorted(trace._list, key=lambda x:x.get('time:timestamp'))
        events = [e for e in events if e.get('lifecycle:transition').upper() == 'COMPLETE' ]

        caseid = trace.attributes.get('concept:name')
        amount_req = trace.attributes.get('AMOUNT_REQ')

        trace_events = [{ 
            'm_case_id': caseid,
            'm_activity': e.get('concept:name'), 
            'm_timestamp': e.get('time:timestamp').strftime("%Y-%m-%d %H:%M:%S"),

            'trace_amount_req': amount_req,        
            'org_resource': e.get('org:resource'), 

            'lifecycle_transition':  e.get('lifecycle:transition'),
            'process_type': e.get('concept:name')[0].upper()
            } for e in events ] 

        trace_events = drop_consecutive_repeating_trace_events(trace_events)
        mark_start_end_for_sorted_trace(trace_events)
        add_delta_seconds_since_first_event_to_trace_events(trace_events)
        
        final_events += trace_events
    
    for e in final_events:
        assert len(e['m_timestamp']) == LEN_TIMESTAMP

    return sorted(final_events, key=lambda x:x['m_timestamp'])


def load_Helpdesk(use_cache=False):
    with open(DATA_RAW_DIR+'/helpdesk/finale.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')
        headers = next(csv_reader)
        headers = [h.lower().replace(' ','_') for h in headers]
        events = [ dict(zip(headers, row_values)) for row_values in csv_reader ]
    
    for idx, event in enumerate(events):
        event["m_case_id"] = event["case_id"]
        event["m_timestamp"] = event["complete_timestamp"].replace("/",'-')[:LEN_TIMESTAMP]
        event["m_activity"] = event["activity"]

    traces = groupby( lambda x:x['m_case_id'], sorted(events, key=lambda x:x['complete_timestamp']) )
    final_events = []
    for _trace in traces.values() :
        trace = drop_consecutive_repeating_trace_events(_trace)
        mark_start_end_for_sorted_trace(trace)
        add_delta_seconds_since_first_event_to_trace_events(trace)
        final_events += trace
    
    for e in final_events:
        assert len(e['m_timestamp']) == LEN_TIMESTAMP

    return sorted(final_events, key=lambda x:x['complete_timestamp'])


def load_Sepsis(use_cache=False):
    
    xes = XESData(DATA_RAW_DIR+'/sepsis/Sepsis Cases - Event Log.xes', DATA_OUTPUT_DIR+'/tmp.sepsis.pickle',use_cache=use_cache)
    traces = xes.traces

    context_data = [
     'Hypoxie',
     'InfectionSuspected',
     'DiagnosticLiquor',
     'SIRSCritHeartRate',
     'Age',
     'DisfuncOrg',
     'LacticAcid',
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
     'CRP',
     'DiagnosticUrinarySediment',
     'Leucocytes',

     'lifecycle:transition'
    ]

    final_events = []

    for idx, trace in enumerate(traces):
        # We sort using the full fractional seconds available but bellow
        # cut the fractional seconds because python's sorting is stable. 
        events = sorted(trace._list, key=lambda x:x.get('time:timestamp'))
        events = [e for e in events if e.get('lifecycle:transition').upper() == 'COMPLETE' ]

        caseid = trace.attributes.get('concept:name')
        
        trace_events = []
        for e in events:
            _e = { 
                'm_case_id': caseid,
                'm_activity': e.get('concept:name'), 
                'm_timestamp': e.get('time:timestamp').strftime("%Y-%m-%d %H:%M:%S"),
            }
            for attr in context_data:
                _e[attr] = e.get(attr, None)
            trace_events.append(_e)

        trace_events = drop_consecutive_repeating_trace_events(trace_events)
        mark_start_end_for_sorted_trace(trace_events)
        add_delta_seconds_since_first_event_to_trace_events(trace_events)
        
        final_events += trace_events
    
    for e in final_events:
        assert len(e['m_timestamp']) == LEN_TIMESTAMP

    return sorted(final_events, key=lambda x:x['m_timestamp'])

    

# Public API
def HelpdeskEventLogs(use_cache=False):
    return EventLogLoader(load_Helpdesk(use_cache))

def BPI2012EventLogs(use_cache=False):
    return EventLogLoader(load_BPI2012(use_cache))

def SepsisEventLogs(use_cache=False):
    return EventLogLoader(load_Sepsis(use_cache))
