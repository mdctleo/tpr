import requests
import time
import threading
import random
from tqdm import tqdm
import argparse
import csv
import time
import random


def generate_evenly_spaced_timestamps(start_ns, end_ns, num_batches):
    interval_ns = (end_ns - start_ns) // num_batches
    return [start_ns + i * interval_ns for i in range(num_batches)]

def regular_client(server_url, start_offset=0):
    # Generate a fake timestamp
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    file, writer = init_record("dos_reg.csv")

    step_size = 5
    for second in tqdm(range(0, int(total_seconds), step_size), total=int(total_seconds)//step_size, desc="Sending regular requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(1, 5)

        for j in range(num_requests_in_batch):
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
            record_dos(file, writer, current_timestamp_ns)
    
        time.sleep(0.05)


def dos_client(server_url, start_offset=0):
    # Do regular client stuff first
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    file, writer = init_record("dos_adv.csv")

    step_size = 5
    for second in tqdm(range(0, int(total_seconds), step_size), total=int(total_seconds)//step_size, desc="Sending regular requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(1, 5)

        for j in range(num_requests_in_batch):
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
            record_dos(file, writer, current_timestamp_ns)
    
        time.sleep(0.05)


    # now we sprinkle in the adversarial requests
    start_ns = time.time_ns() + start_offset * 3600 * 1e9 + 2200 * 1e9
    end_ns = start_ns + 300 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    step_size = 5
    for second in tqdm(range(0, int(total_seconds), step_size), total=int(total_seconds) // step_size, desc="Sending DOS requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(10, 50)

        for j in range(num_requests_in_batch):
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
            record_dos(file, writer, current_timestamp_ns)
    
        time.sleep(0.05)

    # def send_request(second):
    #     try:            
    #         current_timestamp_ns = start_ns + second * 1e9
    #         num_request_in_batch = random.randint(10, 50)
    #         for j in range(num_request_in_batch):
    #             # Send GET request with fake timestamp
    #             headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
    #             response = requests.get(server_url, headers=headers)
    #             print(f"GET request: {response.status_code}")

    #             # Wait for the specified interval before sending the next request
    #     except requests.exceptions.RequestException as e:
    #         print(f"Request failed: {e}")

    # threads = []
    # for second in range(0, int(total_seconds), 5):
    #     time.sleep(0.05)
    #     thread = threading.Thread(target=send_request, args=(second,))
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()
def regular_bruteforce_client(server_url, start_offset=0):
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    file, writer = init_record("brute_reg.csv")

    step_size = 300

    for second in tqdm(range(0, int(total_seconds), step_size), total=int(total_seconds)//step_size, desc="Sending bruteforce requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = 1

        for j in range(num_requests_in_batch):
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
            record_brute_force(file, writer, current_timestamp_ns)
    
        time.sleep(0.05)

def bruteforce_client(server_url, start_offset=0):
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    file, writer = init_record("brute_adv.csv")

    step_size = 10

    for second in tqdm(range(0, int(total_seconds), step_size), total=int(total_seconds)//step_size, desc="Sending bruteforce requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(1, 1)

        for j in range(num_requests_in_batch):
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
            record_brute_force(file, writer, current_timestamp_ns)
    
        time.sleep(0.05)


    # for second in tqdm(range(0, int(total_seconds), 5), total=int(total_seconds), desc="Sending bruteforce requests"):
    #     current_timestamp_ns = start_ns + second * 1e9

    #     num_requests_in_batch = random.randint(2, 10)

    #     for j in range(num_requests_in_batch):
    #         # Send GET request with fake timestamp
    #         headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
    #         response = requests.get(server_url, headers=headers)
    
    #     time.sleep(0.05)

def regular_ticket_client(server_url, start_offset=0):
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    request_clients = [{} for _ in range(1)]
    for request_client in request_clients:
        # if random.random() < 0.5:
        #     request_client['ticket_timestamp'] = (time.time() + start_offset + 600) * 1e9
        # else:
        #     request_client['ticket_timestamp'] = (time.time() + start_offset - 60) * 1e9

        request_client['ticket_timestamp'] = (time.time() + start_offset + 600) * 1e9

        request_client["requests"] = []

    file, writer = init_record("ticket_reg.csv")

    for second in tqdm(range(0, int(total_seconds), 5), total=int(total_seconds), desc="Sending regular requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(1, 5)

        for j in range(num_requests_in_batch):
            client = random.choice(request_clients)

            client["requests"].append(current_timestamp_ns)
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
    
    for client in request_clients:
        for request_timestamp in client["requests"]:
            while not access_service(request_timestamp, client["ticket_timestamp"], file, writer):
                client["ticket_timestamp"] = get_ticket(request_timestamp, file, writer)

def adversary_ticket_client(server_url, start_offset=0):
    start_ns = time.time_ns() + start_offset * 3600 * 1e9
    end_ns = start_ns + 3600 * 1e9

    total_seconds = (end_ns - start_ns) / 1e9

    print("total seconds: ", total_seconds)

    request_clients = [{} for _ in range(1)]
    for i in range(1):
        # if i < 100:
        #     request_clients[i]['ticket_timestamp'] = float('inf')
        # elif random.random() < 0.5:
        #     request_clients[i]['ticket_timestamp'] = (time.time() + start_offset + 600) * 1e9
        # else:
        #     request_clients[i]['ticket_timestamp'] = (time.time() + start_offset - 60) * 1e9
        request_clients[i]['ticket_timestamp'] = (time.time() + start_offset + 600) * 1e9

        request_clients[i]["requests"] = []

    file, writer = init_record("ticket_adv.csv")
    for second in tqdm(range(0, int(total_seconds), 5), total=int(total_seconds), desc="Sending regular requests"):
        current_timestamp_ns = start_ns + second * 1e9

        num_requests_in_batch = random.randint(1, 5)

        for j in range(num_requests_in_batch):
            client = random.choice(request_clients)

            client["requests"].append(current_timestamp_ns)
            # Send GET request with fake timestamp
            # headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
            # response = requests.get(server_url, headers=headers)
    
    for client in request_clients:
        for i in range(len(client["requests"])):
            request_timestamp = client["requests"][i]
            if i >= len(client["requests"]) // 2:
                client["ticket_timestamp"] = float("inf")

            while not access_service(request_timestamp, client["ticket_timestamp"], file, writer):
                client["ticket_timestamp"] = get_ticket(request_timestamp, file, writer)


def get_ticket(fake_timestamp_ns, file, writer):
    new_ticket_timestamp = fake_timestamp_ns + 600 * 1e9
    record_ticket(file, writer, fake_timestamp_ns, "get_ticket")
    return new_ticket_timestamp


def access_service(fake_timestamp_ns, ticket_timestamp, file, writer):
    if ticket_timestamp is None or fake_timestamp_ns > ticket_timestamp:
        return False

    # record some things here
    record_ticket(file, writer, fake_timestamp_ns, "access_service")
    return True

# def get_ticket(server_url, current_timestamp_ns):
#     headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
#     response = requests.post(server_url, headers=headers)
#     print(f"POST request: {response.status_code}")

#     return response


# def access_service(server_url, current_timestamp_ns):
#     headers = {'X-Fake-Timestamp': str(current_timestamp_ns)}
#     response = requests.get(server_url, headers=headers)
#     print(f"GET request: {response.status_code}")

#     return response

def init_record(file_name):
    csv_headers = ['subjectname', 'subject_type', 'objectname', 'object_type', 'syscall', 'timestamp']
    file = open(file_name, 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(csv_headers)
    
    return file, writer

def record_dos(file, writer, fake_timestamp_ns):
    writer = csv.writer(file)
    writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", "fakesyscall", fake_timestamp_ns])
    file.flush()

def record_brute_force(file, writer, fake_timestamp_ns):
    writer = csv.writer(file)
    writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", "fakesyscall", fake_timestamp_ns])
    file.flush()

def record_ticket(file, writer, fake_timestamp_ns, operation):
    writer = csv.writer(file)
    writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", operation, fake_timestamp_ns])
    file.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client configs")
    parser.add_argument("--client_type", type=str, help="Type of experiment to run")
    parser.add_argument("--start_offset", type=int, help="Start offset for the experiment")
    args = parser.parse_args()

    server_url = 'http://localhost:9500'
    # simulate regular client
    if args.client_type == "dos_reg":
        regular_client(server_url, start_offset=args.start_offset)
    elif args.client_type == "dos_adv":
        dos_client(server_url, start_offset=args.start_offset)
    elif args.client_type == "brute_force_reg":
        regular_bruteforce_client(server_url, start_offset=args.start_offset)
    elif args.client_type == "brute_force_adv":
        bruteforce_client(server_url, start_offset=args.start_offset)
    elif args.client_type == "regular_ticket":
        regular_ticket_client(server_url, start_offset=args.start_offset)
    elif args.client_type == "adversary_ticket":
        adversary_ticket_client(server_url, start_offset=args.start_offset)

    # Simulate brute force client
    # bruteforce_client(server_url, start_offset=4)