#!/usr/bin/env python3
"""
hv_attacks_config.py — Auto-generated per-attack constants for all 43 HyperVision attacks.
Generated from config.json files by process_hypervision.py with adaptive snapshot delta.
Attacks with doe < 5s use 1s snapshot delta; others use 5s.
"""

import os

HV_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    '1-DATA_PROCESSING', 'hypervision', 'output'
)

HV_ATTACKS = {
    'ackport': {
        'date_of_evil': 17857399,  # 17.857399s in microseconds
        'time_range': 52301926,  # 52.301926s in microseconds
        'unique_ips': 42762,
        'total_flows': 96607,
        'attack_flows': 49,
        'benign_flows': 96558,
        'trainwin': 3,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'charrdos': {
        'date_of_evil': 5638164,  # 5.638164s in microseconds
        'time_range': 41861657,  # 41.861657s in microseconds
        'unique_ips': 138774,
        'total_flows': 191542,
        'attack_flows': 209,
        'benign_flows': 191333,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'cldaprdos': {
        'date_of_evil': 3001039,  # 3.001039s in microseconds
        'time_range': 38879268,  # 38.879268s in microseconds
        'unique_ips': 139872,
        'total_flows': 192642,
        'attack_flows': 1309,
        'benign_flows': 191333,
        'trainwin': 3,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'crossfirela': {
        'date_of_evil': 9296757,  # 9.296757s in microseconds
        'time_range': 37698289,  # 37.698289s in microseconds
        'unique_ips': 178683,
        'total_flows': 364045,
        'attack_flows': 291762,
        'benign_flows': 72283,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'crossfiremd': {
        'date_of_evil': 864,  # 0.000864s in microseconds
        'time_range': 28582316,  # 28.582316s in microseconds
        'unique_ips': 178606,
        'total_flows': 310286,
        'attack_flows': 238011,
        'benign_flows': 72275,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'crossfiresm': {
        'date_of_evil': 29069673,  # 29.069673s in microseconds
        'time_range': 45153378,  # 45.153378s in microseconds
        'unique_ips': 178581,
        'total_flows': 290323,
        'attack_flows': 74979,
        'benign_flows': 215344,
        'trainwin': 5,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'dns_lrscan': {
        'date_of_evil': 723,  # 0.000723s in microseconds
        'time_range': 29427720,  # 29.427720s in microseconds
        'unique_ips': 133226,
        'total_flows': 583881,
        'attack_flows': 502582,
        'benign_flows': 81299,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'dnsrdos': {
        'date_of_evil': 5954163,  # 5.954163s in microseconds
        'time_range': 40774304,  # 40.774304s in microseconds
        'unique_ips': 138774,
        'total_flows': 191542,
        'attack_flows': 209,
        'benign_flows': 191333,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'dnsscan': {
        'date_of_evil': 629034,  # 0.629034s in microseconds
        'time_range': 30736900,  # 30.736900s in microseconds
        'unique_ips': 105867,
        'total_flows': 159705,
        'attack_flows': 63174,
        'benign_flows': 96531,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'http_lrscan': {
        'date_of_evil': 586,  # 0.000586s in microseconds
        'time_range': 30055323,  # 30.055323s in microseconds
        'unique_ips': 183130,
        'total_flows': 1150550,
        'attack_flows': 1078291,
        'benign_flows': 72259,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'httpscan': {
        'date_of_evil': 665432,  # 0.665432s in microseconds
        'time_range': 32815464,  # 32.815464s in microseconds
        'unique_ips': 105665,
        'total_flows': 162669,
        'attack_flows': 64030,
        'benign_flows': 98639,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'httpsscan': {
        'date_of_evil': 1063050,  # 1.063050s in microseconds
        'time_range': 34518469,  # 34.518469s in microseconds
        'unique_ips': 164676,
        'total_flows': 224894,
        'attack_flows': 128368,
        'benign_flows': 96526,
        'trainwin': 1,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'icmp_lrscan': {
        'date_of_evil': 3813947,  # 3.813947s in microseconds
        'time_range': 30289322,  # 30.289322s in microseconds
        'unique_ips': 125261,
        'total_flows': 1932185,
        'attack_flows': 1762376,
        'benign_flows': 169809,
        'trainwin': 3,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'icmpscan': {
        'date_of_evil': 649951,  # 0.649951s in microseconds
        'time_range': 34713103,  # 34.713103s in microseconds
        'unique_ips': 181575,
        'total_flows': 249345,
        'attack_flows': 137201,
        'benign_flows': 112144,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'icmpsdos': {
        'date_of_evil': 4120981,  # 4.120982s in microseconds
        'time_range': 38394146,  # 38.394146s in microseconds
        'unique_ips': 142271,
        'total_flows': 195042,
        'attack_flows': 104460,
        'benign_flows': 90582,
        'trainwin': 4,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'ipidaddr': {
        'date_of_evil': 100348,  # 0.100348s in microseconds
        'time_range': 113071679,  # 113.071679s in microseconds
        'unique_ips': 140178,
        'total_flows': 192945,
        'attack_flows': 1656,
        'benign_flows': 191289,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'ipidport': {
        'date_of_evil': 25453694,  # 25.453694s in microseconds
        'time_range': 114565014,  # 114.565014s in microseconds
        'unique_ips': 138584,
        'total_flows': 191348,
        'attack_flows': 100758,
        'benign_flows': 90590,
        'trainwin': 5,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'lrtcpdos02': {
        'date_of_evil': 18762523,  # 18.762523s in microseconds
        'time_range': 63023387,  # 63.023387s in microseconds
        'unique_ips': 42731,
        'total_flows': 96534,
        'attack_flows': 49,
        'benign_flows': 96485,
        'trainwin': 3,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'lrtcpdos05': {
        'date_of_evil': 14013355,  # 14.013355s in microseconds
        'time_range': 154826855,  # 154.826855s in microseconds
        'unique_ips': 42765,
        'total_flows': 96598,
        'attack_flows': 78,
        'benign_flows': 96520,
        'trainwin': 2,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'lrtcpdos10': {
        'date_of_evil': 17787353,  # 17.787353s in microseconds
        'time_range': 72756760,  # 72.756760s in microseconds
        'unique_ips': 42758,
        'total_flows': 96591,
        'attack_flows': 78,
        'benign_flows': 96513,
        'trainwin': 3,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'memcachedrdos': {
        'date_of_evil': 2104133,  # 2.104133s in microseconds
        'time_range': 35231607,  # 35.231607s in microseconds
        'unique_ips': 138734,
        'total_flows': 191502,
        'attack_flows': 169,
        'benign_flows': 191333,
        'trainwin': 2,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'netbios_lrscan': {
        'date_of_evil': 758,  # 0.000758s in microseconds
        'time_range': 29039091,  # 29.039091s in microseconds
        'unique_ips': 133752,
        'total_flows': 598413,
        'attack_flows': 517114,
        'benign_flows': 81299,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'ntprdos': {
        'date_of_evil': 4276528,  # 4.276528s in microseconds
        'time_range': 38138714,  # 38.138714s in microseconds
        'unique_ips': 139223,
        'total_flows': 191992,
        'attack_flows': 659,
        'benign_flows': 191333,
        'trainwin': 4,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'ntpscan': {
        'date_of_evil': 1175379,  # 1.175379s in microseconds
        'time_range': 30160244,  # 30.160244s in microseconds
        'unique_ips': 108846,
        'total_flows': 162685,
        'attack_flows': 66153,
        'benign_flows': 96532,
        'trainwin': 1,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'rdp_lrscan': {
        'date_of_evil': 3305457,  # 3.305457s in microseconds
        'time_range': 29915425,  # 29.915425s in microseconds
        'unique_ips': 126195,
        'total_flows': 1478049,
        'attack_flows': 1308240,
        'benign_flows': 169809,
        'trainwin': 3,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'riprdos': {
        'date_of_evil': 9048667,  # 9.048667s in microseconds
        'time_range': 42322045,  # 42.322045s in microseconds
        'unique_ips': 139073,
        'total_flows': 191842,
        'attack_flows': 509,
        'benign_flows': 191333,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'rstsdos': {
        'date_of_evil': 1656537,  # 1.656537s in microseconds
        'time_range': 34834679,  # 34.834679s in microseconds
        'unique_ips': 171245,
        'total_flows': 224019,
        'attack_flows': 133437,
        'benign_flows': 90582,
        'trainwin': 1,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'smtp_lrscan': {
        'date_of_evil': 731,  # 0.000731s in microseconds
        'time_range': 28546783,  # 28.546783s in microseconds
        'unique_ips': 131157,
        'total_flows': 309902,
        'attack_flows': 228603,
        'benign_flows': 81299,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'snmp_lrscan': {
        'date_of_evil': 709,  # 0.000709s in microseconds
        'time_range': 28488621,  # 28.488621s in microseconds
        'unique_ips': 136243,
        'total_flows': 276731,
        'attack_flows': 195426,
        'benign_flows': 81305,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'sqlscan': {
        'date_of_evil': 23390128,  # 23.390128s in microseconds
        'time_range': 57479385,  # 57.479385s in microseconds
        'unique_ips': 120385,
        'total_flows': 176523,
        'attack_flows': 77751,
        'benign_flows': 98772,
        'trainwin': 4,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'ssdprdos': {
        'date_of_evil': 6204944,  # 6.204944s in microseconds
        'time_range': 39924996,  # 39.924996s in microseconds
        'unique_ips': 139872,
        'total_flows': 192811,
        'attack_flows': 1478,
        'benign_flows': 191333,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'ssh_lrscan': {
        'date_of_evil': 72680,  # 0.072680s in microseconds
        'time_range': 36475661,  # 36.475661s in microseconds
        'unique_ips': 168914,
        'total_flows': 264489,
        'attack_flows': 94715,
        'benign_flows': 169774,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'sshpwdla': {
        'date_of_evil': 34344767,  # 34.344767s in microseconds
        'time_range': 53002363,  # 53.002363s in microseconds
        'unique_ips': 118624,
        'total_flows': 171031,
        'attack_flows': 1200,
        'benign_flows': 169831,
        'trainwin': 6,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'sshpwdmd': {
        'date_of_evil': 801,  # 0.000801s in microseconds
        'time_range': 29894796,  # 29.894796s in microseconds
        'unique_ips': 118520,
        'total_flows': 170417,
        'attack_flows': 89110,
        'benign_flows': 81307,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'sshpwdsm': {
        'date_of_evil': 56250950,  # 56.250950s in microseconds
        'time_range': 73736872,  # 73.736872s in microseconds
        'unique_ips': 118482,
        'total_flows': 169936,
        'attack_flows': 100,
        'benign_flows': 169836,
        'trainwin': 11,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'sshscan': {
        'date_of_evil': 651474,  # 0.651474s in microseconds
        'time_range': 127699697,  # 127.699697s in microseconds
        'unique_ips': 398104,
        'total_flows': 462499,
        'attack_flows': 353890,
        'benign_flows': 108609,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'synsdos': {
        'date_of_evil': 9354909,  # 9.354909s in microseconds
        'time_range': 42228817,  # 42.228817s in microseconds
        'unique_ips': 204094,
        'total_flows': 256877,
        'attack_flows': 166295,
        'benign_flows': 90582,
        'trainwin': 1,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'telnet_lrscan': {
        'date_of_evil': 3201736,  # 3.201736s in microseconds
        'time_range': 27464412,  # 27.464412s in microseconds
        'unique_ips': 126494,
        'total_flows': 1406117,
        'attack_flows': 1236308,
        'benign_flows': 169809,
        'trainwin': 3,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'telnetpwdla': {
        'date_of_evil': 35580603,  # 35.580603s in microseconds
        'time_range': 47767126,  # 47.767126s in microseconds
        'unique_ips': 118520,
        'total_flows': 170417,
        'attack_flows': 600,
        'benign_flows': 169817,
        'trainwin': 7,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'telnetpwdmd': {
        'date_of_evil': 35141401,  # 35.141401s in microseconds
        'time_range': 47126807,  # 47.126807s in microseconds
        'unique_ips': 118470,
        'total_flows': 170117,
        'attack_flows': 300,
        'benign_flows': 169817,
        'trainwin': 7,
        'pikachu_snapshot_delta': 5.0,  # seconds
    },
    'telnetpwdsm': {
        'date_of_evil': 721,  # 0.000721s in microseconds
        'time_range': 31016758,  # 31.016758s in microseconds
        'unique_ips': 118444,
        'total_flows': 169961,
        'attack_flows': 88654,
        'benign_flows': 81307,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'udpsdos': {
        'date_of_evil': 989619,  # 0.989619s in microseconds
        'time_range': 33690900,  # 33.690900s in microseconds
        'unique_ips': 171268,
        'total_flows': 224042,
        'attack_flows': 133460,
        'benign_flows': 90582,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
    'vlc_lrscan': {
        'date_of_evil': 723,  # 0.000723s in microseconds
        'time_range': 28153218,  # 28.153218s in microseconds
        'unique_ips': 131942,
        'total_flows': 466861,
        'attack_flows': 385562,
        'benign_flows': 81299,
        'trainwin': 0,
        'pikachu_snapshot_delta': 1,  # seconds
    },
}

HV_ATTACK_NAMES = frozenset(HV_ATTACKS.keys())  # 43 attacks

# Attacks eligible for KDE (trainwin >= 1)
HV_KDE_ELIGIBLE = frozenset([
    'ackport',
    'charrdos',
    'cldaprdos',
    'crossfirela',
    'crossfiresm',
    'dnsrdos',
    'httpsscan',
    'icmp_lrscan',
    'icmpsdos',
    'ipidport',
    'lrtcpdos02',
    'lrtcpdos05',
    'lrtcpdos10',
    'memcachedrdos',
    'ntprdos',
    'ntpscan',
    'rdp_lrscan',
    'riprdos',
    'rstsdos',
    'sqlscan',
    'ssdprdos',
    'sshpwdla',
    'sshpwdsm',
    'synsdos',
    'telnet_lrscan',
    'telnetpwdla',
    'telnetpwdmd',
])  # 27 attacks

