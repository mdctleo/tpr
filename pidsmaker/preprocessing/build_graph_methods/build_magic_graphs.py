import json
import os
from datetime import datetime, timedelta

import networkx as nx
import torch

from pidsmaker.config import get_darpa_tc_node_feats_from_cfg
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    init_database_connection,
    log,
    log_start,
    ns_time_to_datetime_US,
)


def get_node_list(cur, cfg):
    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)

    uuid2idx = {}
    uuid2type = {}
    uuid2name = {}
    hash2uuid = {}

    # netflow
    sql = "select * from netflow_node_table;"
    cur.execute(sql)

    while True:
        records = cur.fetchmany(1000)
        if not records:
            break
        # node_uuid | hash_id | src_addr | src_port | dst_addr | dst_port | index_id
        for i in records:
            attrs = {
                "type": "netflow",
                "local_ip": str(i[2]),
                "local_port": str(i[3]),
                "remote_ip": str(i[4]),
                "remote_port": str(i[5]),
            }
            node_uuid = str(i[0])
            hash_id = str(i[1])
            index_id = int(i[-1])

            features_used = []
            for label_used in node_label_features["netflow"]:
                features_used.append(attrs[label_used])
            label_str = " ".join(features_used)

            uuid2idx[node_uuid] = index_id
            uuid2type[node_uuid] = attrs["type"]
            uuid2name[node_uuid] = label_str
            hash2uuid[hash_id] = node_uuid

    del records

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    while True:
        records = cur.fetchmany(1000)
        if not records:
            break
        # node_uuid | hash_id | path | cmd | index_id
        for i in records:
            attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
            node_uuid = str(i[0])
            hash_id = str(i[1])
            index_id = int(i[-1])
            features_used = []
            for label_used in node_label_features["subject"]:
                features_used.append(attrs[label_used])
            label_str = " ".join(features_used)

            uuid2idx[node_uuid] = index_id
            uuid2type[node_uuid] = attrs["type"]
            uuid2name[node_uuid] = label_str
            hash2uuid[hash_id] = node_uuid

    del records

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    while True:
        records = cur.fetchmany(1000)
        if not records:
            break
        # node_uuid | hash_id | path | index_id
        for i in records:
            attrs = {"type": "file", "path": str(i[2])}
            node_uuid = str(i[0])
            hash_id = str(i[1])
            index_id = int(i[-1])
            features_used = []
            for label_used in node_label_features["file"]:
                features_used.append(attrs[label_used])
            label_str = " ".join(features_used)

            uuid2idx[node_uuid] = index_id
            uuid2type[node_uuid] = attrs["type"]
            uuid2name[node_uuid] = label_str
            hash2uuid[hash_id] = node_uuid

    del records

    return uuid2idx, uuid2type, uuid2name, hash2uuid


def generate_timestamps(start_time, end_time, interval_minutes):
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    timestamps = []
    current_time = start
    while current_time <= end:
        timestamps.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
        current_time += timedelta(minutes=interval_minutes)
    timestamps.append(end)
    return timestamps


def get_attack_time_windows_for_day(cfg, day):
    """
    Returns a list of attack time windows (as ns timestamps) for a given day.
    Each entry is (attack_name, start_ns, end_ns).
    """
    attack_windows = []
    year_month = cfg.dataset.year_month
    day_str = f"{year_month}-{day:02d}"
    
    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack_name = attack_tuple[0]
        start_str = attack_tuple[1]  # e.g., "2017-07-05 09:47:00"
        end_str = attack_tuple[2]
        
        # Check if this attack is on the current day
        if start_str.startswith(day_str):
            start_ns = datetime_to_ns_time_US(start_str)
            end_ns = datetime_to_ns_time_US(end_str)
            attack_windows.append((attack_name, start_ns, end_ns))
    
    return attack_windows


def filter_events_for_attack(events_list, target_attack_window, all_attack_windows):
    """
    Filter events to include:
    1. All traffic outside any attack window (benign traffic)
    2. Traffic during the target attack window only
    
    Args:
        events_list: List of event tuples
        target_attack_window: (attack_name, start_ns, end_ns) for the target attack
        all_attack_windows: List of all attack windows for this day
    
    Returns:
        Filtered list of events
    """
    target_name, target_start, target_end = target_attack_window
    
    # Build a list of "other attack" time ranges to exclude
    other_attack_ranges = []
    for attack_name, start_ns, end_ns in all_attack_windows:
        if attack_name != target_name:
            other_attack_ranges.append((start_ns, end_ns))
    
    filtered_events = []
    for event_tuple in events_list:
        timestamp = event_tuple[-2]  # timestamp_rec is the 7th element (index 6 or -2)
        
        # Check if this event is during another attack's window
        in_other_attack = False
        for other_start, other_end in other_attack_ranges:
            if other_start <= timestamp <= other_end:
                in_other_attack = True
                break
        
        # Include event if it's NOT during another attack's window
        if not in_other_attack:
            filtered_events.append(event_tuple)
    
    return filtered_events


def generate_per_attack_test_graphs(cur, uuid2type, graph_out_dir, hash2uuid, cfg, test_days):
    """
    Generate per-attack test graphs for CIC_IDS_2017_PER_ATTACK config.
    Each test graph contains full day benign traffic + only one specific attack.
    """
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map(cfg=cfg)
    include_edge_type = rel2id
    node_type_dict = ntype2id
    
    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    
    window_size_in_sec = cfg.construction.time_window_size * 60_000_000_000
    BATCH = 1024
    
    # Get mapping from test_file to attack index (stored as list of tuples for YACS compatibility)
    test_file_to_attack_idx_list = getattr(cfg.dataset, 'test_file_to_attack_idx', [])
    test_file_to_attack_idx = dict(test_file_to_attack_idx_list) if test_file_to_attack_idx_list else {}
    
    for test_file in cfg.dataset.test_files:
        # Parse day from test_file (e.g., "graph_5_dos_slowloris" -> 5)
        day = int(test_file.split("_")[1])
        
        if day not in test_days:
            continue
        
        # Get attack index for this test file
        attack_idx = test_file_to_attack_idx.get(test_file)
        if attack_idx is None:
            log(f"Warning: No attack mapping for {test_file}, skipping")
            continue
        
        # Get target attack time window
        attack_tuple = cfg.dataset.attack_to_time_window[attack_idx]
        attack_name = attack_tuple[0]
        target_attack = (attack_name, 
                        datetime_to_ns_time_US(attack_tuple[1]), 
                        datetime_to_ns_time_US(attack_tuple[2]))
        
        # Get all attack windows for this day
        all_attack_windows = get_attack_time_windows_for_day(cfg, day)
        
        log(f"Generating per-attack graph for {test_file} (attack: {attack_name})")
        
        # Query full day's events
        date_start = cfg.dataset.year_month + "-" + str(day) + " 00:00:00"
        date_stop = cfg.dataset.year_month + "-" + str(day + 1) + " 00:00:00"
        start_ns_timestamp = datetime_to_ns_time_US(date_start)
        end_ns_timestamp = datetime_to_ns_time_US(date_stop)
        
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_ns_timestamp, end_ns_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        
        if len(events) == 0:
            log(f"No events found for day {day}")
            continue
        
        # Filter events to include edge types
        events_list = []
        for (
            src_node, src_index_id, operation, dst_node, dst_index_id,
            event_uuid, timestamp_rec, _id,
        ) in events:
            if operation in include_edge_type:
                event_tuple = (
                    src_node, src_index_id, operation, dst_node, dst_index_id,
                    event_uuid, timestamp_rec, _id,
                )
                events_list.append(event_tuple)
        
        if len(events_list) == 0:
            log(f"No valid events for day {day}")
            continue
        
        # Filter events: keep benign + target attack only
        filtered_events = filter_events_for_attack(events_list, target_attack, all_attack_windows)
        
        log(f"  Original events: {len(events_list)}, Filtered: {len(filtered_events)}")
        
        if len(filtered_events) == 0:
            log(f"  Warning: No events after filtering for {test_file}")
            continue
        
        # Create time-windowed graphs from filtered events
        start_time = filtered_events[0][-2]
        temp_list = []
        last_batch = False
        
        for batch_edges in get_batches(filtered_events, BATCH):
            for j in batch_edges:
                temp_list.append(j)
            
            if (len(batch_edges) < BATCH) or (temp_list[-1] == filtered_events[-1]):
                last_batch = True
            
            if (batch_edges[-1][-2] > start_time + window_size_in_sec) or last_batch:
                time_interval = (
                    ns_time_to_datetime_US(start_time)
                    + "~"
                    + ns_time_to_datetime_US(batch_edges[-1][-2])
                )
                
                log(f"  Creating time window graph for {time_interval}")
                
                g = nx.DiGraph()
                node_visited = set()
                
                for event_tuple in filtered_events:
                    (
                        src_node, src_index_id, operation, dst_node, dst_index_id,
                        event_uuid, timestamp_rec, _id,
                    ) = event_tuple
                    if src_index_id not in node_visited:
                        g.add_node(
                            int(src_index_id),
                            type=node_type_dict[uuid2type[hash2uuid[src_node]]],
                        )
                        node_visited.add(src_index_id)
                    if dst_index_id not in node_visited:
                        g.add_node(
                            int(dst_index_id),
                            type=node_type_dict[uuid2type[hash2uuid[dst_node]]],
                        )
                        node_visited.add(dst_index_id)
                    if not g.has_edge(int(src_index_id), int(dst_index_id)):
                        g.add_edge(
                            int(src_index_id),
                            int(dst_index_id),
                            type=include_edge_type[operation],
                        )
                
                # Save to per-attack folder
                date_dir = f"{graph_out_dir}/{test_file}/"
                os.makedirs(date_dir, exist_ok=True)
                graph_name = f"{date_dir}/{time_interval}"
                
                print(f"Saving graph for {test_file}: {time_interval}")
                torch.save(g, graph_name)
                
                start_time = batch_edges[-1][-2]
                temp_list.clear()
                
                # For unit tests, only create one TW per attack
                if cfg._test_mode:
                    break
    
    return


def generate_graphs(cur, uuid2type, graph_out_dir, hash2uuid, cfg):
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map(cfg=cfg)
    include_edge_type = rel2id
    node_type_dict = ntype2id

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]

    # Check if per-attack test graphs are enabled
    per_attack_test_graphs = getattr(cfg.dataset, 'per_attack_test_graphs', False)
    
    # Get test day numbers (to skip in main loop if per_attack_test_graphs is enabled)
    test_days = set()
    if per_attack_test_graphs:
        for test_file in cfg.dataset.test_files:
            # Parse day from test_file (e.g., "graph_5_dos_slowloris" -> 5)
            parts = test_file.split("_")
            if len(parts) >= 2:
                try:
                    test_days.add(int(parts[1]))
                except ValueError:
                    pass
        log(f"Per-attack test graphs enabled. Test days to skip in main loop: {test_days}")

    # In test mode, we ensure to get 1 TW in each set
    if cfg._test_mode:
        # Get the day number of the first day in each set
        days = [
            int(days[0].split("_")[-1])
            for days in [cfg.dataset.train_files, cfg.dataset.val_files, cfg.dataset.test_files]
        ]
    else:
        start, end = cfg.dataset.start_end_day_range
        days = range(start, end)

    for day in days:
        # Skip test days if per-attack test graphs will be generated separately
        if per_attack_test_graphs and day in test_days:
            log(f"Skipping day {day} - will generate per-attack graphs instead")
            continue
        
        date_start = cfg.dataset.year_month + "-" + str(day) + " 00:00:00"
        date_stop = cfg.dataset.year_month + "-" + str(day + 1) + " 00:00:00"

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start)
            end_ns_timestamp = datetime_to_ns_time_US(stop)
            sql = """
            select * from event_table
            where
                  timestamp_rec>'%s' and timestamp_rec<'%s'
                   ORDER BY timestamp_rec;
            """ % (start_ns_timestamp, end_ns_timestamp)
            cur.execute(sql)
            events = cur.fetchall()

            if len(events) == 0:
                continue

            events_list = []
            for (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in events:
                if operation in include_edge_type:
                    event_tuple = (
                        src_node,
                        src_index_id,
                        operation,
                        dst_node,
                        dst_index_id,
                        event_uuid,
                        timestamp_rec,
                        _id,
                    )
                    events_list.append(event_tuple)

            start_time = events_list[0][-2]
            temp_list = []
            BATCH = 1024
            window_size_in_sec = cfg.construction.time_window_size * 60_000_000_000

            last_batch = False
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
                    last_batch = True

                if (batch_edges[-1][-2] > start_time + window_size_in_sec) or last_batch:
                    time_interval = (
                        ns_time_to_datetime_US(start_time)
                        + "~"
                        + ns_time_to_datetime_US(batch_edges[-1][-2])
                    )

                    log(f"Start create edge fused time window graph for {time_interval}")

                    g = nx.DiGraph()
                    node_visited = set()

                    for event_tuple in events_list:
                        (
                            src_node,
                            src_index_id,
                            operation,
                            dst_node,
                            dst_index_id,
                            event_uuid,
                            timestamp_rec,
                            _id,
                        ) = event_tuple
                        if src_index_id not in node_visited:
                            g.add_node(
                                int(src_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[src_node]]],
                            )
                            node_visited.add(src_index_id)
                        if dst_index_id not in node_visited:
                            g.add_node(
                                int(dst_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[dst_node]]],
                            )
                            node_visited.add(dst_index_id)
                        if not g.has_edge(int(src_index_id), int(dst_index_id)):
                            g.add_edge(
                                int(src_index_id),
                                int(dst_index_id),
                                type=include_edge_type[operation],
                            )

                    date_dir = f"{graph_out_dir}/graph_{day}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    print(f"Saving graph for {time_interval}")
                    torch.save(g, graph_name)

                    start_time = batch_edges[-1][-2]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break
    
    # Generate per-attack test graphs if enabled
    if per_attack_test_graphs and test_days:
        generate_per_attack_test_graphs(cur, uuid2type, graph_out_dir, hash2uuid, cfg, test_days)
    
    return


def main(cfg):
    log_start(__file__)
    cur, connect = init_database_connection(cfg)
    uuid2idx, uuid2type, uuid2name, hash2uuid = get_node_list(cur=cur, cfg=cfg)

    os.makedirs(cfg.construction._magic_dir, exist_ok=True)
    os.makedirs(cfg.construction._magic_graphs_dir, exist_ok=True)
    file_out_dir = cfg.construction._magic_dir
    graph_out_dir = cfg.construction._magic_graphs_dir

    with open(os.path.join(file_out_dir, "names.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2name, fw)
    with open(os.path.join(file_out_dir, "types.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2type, fw)

    generate_graphs(
        cur=cur, uuid2type=uuid2type, graph_out_dir=graph_out_dir, hash2uuid=hash2uuid, cfg=cfg
    )

    del uuid2idx, uuid2type, uuid2name, hash2uuid
