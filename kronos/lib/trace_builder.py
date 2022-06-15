from toolz.itertoolz import groupby


def reconstruct_incremental_traces_from_eventlogs(event_logs):
    """ Creates all incremental versions of traces from all event logs.

    Returns an sorted list of dicts: [ { event_sequence : [ <event dict>, ]
                                         next_activity: <activity>  
                                         last_event_timestamp:
                                         activities_set:
                                         case_seq_key:
                                         case_id:

                                       },]

    Assumption: event logs are already sorted by timestamp
    """

    # Event logs should be already ordered
    grpd_eventlogs = groupby(lambda x:x['m_case_id'], event_logs)

    # Order the traces by the timestamp of their respective first event's timestamp 
    grpd_eventlogs = sorted(grpd_eventlogs.values(), key=lambda x:x[0]['m_timestamp'])

    return _create_incremental_process_variations(grpd_eventlogs) 


def _create_incremental_process_variations(grpd_eventlogs):
    """ Expectes a list of event log lists and return a list of 
        lists of all incremental process variations. """
    process_variations = []
    for trace in grpd_eventlogs:
        final = any([event['m_start_end'] == 'end' for event in trace])
        process_variations += _build_incremental_event_sequences_with_metadata(trace, final)
    return process_variations

def _build_incremental_event_sequences_with_metadata(events, full=False):
    """
    Set `full` to true if we have all events for a case.
    Otherwise we ignore the last event as we don't know its successor. 
    E.g.:
     [1,2,3], full=false: 
        1->2  1,2->3

     [1,2,3], full=true:
        1->2  1,2->3  1,2,3->END
   """
    seqs = []
    size = len(events)
    for i in range(size):
        if i+1 == size: 
            if full:
                seqs.append( { 
                    "event_sequence": events[:i+1],
                    "next_activity": 'END',

                    "last_event_timestamp": events[:i+1][-1]['m_timestamp'],
                    "activities_set" : { e['m_activity'] for e in events[:i+1] },
                    "activity_sequence_tuple" : tuple([e['m_activity'] for e in events[:i+1]]),
                    'case_seq_key': build_case_sequence_key(events[:i+1]),
                    'case_id': events[0]['m_case_id']
                })
            else:
                break
        else:
            seqs.append( { 
                "event_sequence": events[:i+1],
                "next_activity": events[i+1]['m_activity'],

                "last_event_timestamp": events[:i+1][-1]['m_timestamp'],
                "activities_set" : { e['m_activity'] for e in events[:i+1] },
                "activity_sequence_tuple" : tuple([e['m_activity'] for e in events[:i+1]]),
                'case_seq_key': build_case_sequence_key(events[:i+1]),
                'case_id': events[0]['m_case_id']
            }) 
    return seqs

def build_case_sequence_key(events):
    case_id = events[0]['m_case_id']
    acts = [ e['m_activity'] for e in events ]
    return ';;'.join([case_id, '::'.join(acts)])
