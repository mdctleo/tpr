import http.server
import logging
from datetime import datetime
import time
import csv
import json
import argparse

# Configure logging

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, csv_file="", **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_file = csv_file
        csv_headers = ['subjectname', 'subject_type', 'objectname', 'object_type', 'syscall', 'timestamp']
        logging.basicConfig(filename='audit.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='a')

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)
    
    def log_request(self, code='-', size='-'):
        fake_timestamp_ns = self.headers.get('X-Fake-Timestamp', None)
        if fake_timestamp_ns is None:
            fake_timestamp_ns = time.time_ns()
        else:
            fake_timestamp_ns = int(float(fake_timestamp_ns))        # Log the request details
            fake_timestamp_s = fake_timestamp_ns / 1e9
            human_readable_timestamp = datetime.fromtimestamp(fake_timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

            print(f"Received request at {human_readable_timestamp}", flush=True)

        # logging.info(f"type=REQUEST msg=audit({fake_timestamp_ns}): method={self.command} path={self.path} version={self.request_version} code={code} size={size}")
        # headers_str = ', '.join([f"{header}={value}" for header, value in self.headers.items()])
        # logging.info(f"type=HEADERS msg=audit({fake_timestamp_ns}): {headers_str} human_readable_timestamp={human_readable_timestamp}")

        # Write the request details to the CSV file
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", "fakesyscall", fake_timestamp_ns])

    def do_GET(self):
        print("received GET request", flush=True)
        # Handle GET request
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"DOS Received!")
        self.log_request(200)


class RequestHandlerBruteForce(http.server.BaseHTTPRequestHandler):
    def __init__():
        super().__init__(self)
        csv_file = "brute_force.csv"
        csv_headers = ['subjectname', 'subject_type', 'objectname', 'object_type', 'syscall', 'timestamp']

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)
    
    def log_request(self, code='-', size='-'):
        fake_timestamp_ns = self.headers.get('X-Fake-Timestamp', None)
        if fake_timestamp_ns is None:
            fake_timestamp_ns = time.time_ns()
        else:
            fake_timestamp_ns = int(float(fake_timestamp_ns))        # Log the request details
            fake_timestamp_s = fake_timestamp_ns / 1e9
            human_readable_timestamp = datetime.fromtimestamp(fake_timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

            print(f"Received request at {human_readable_timestamp}", flush=True)

        # logging.info(f"type=REQUEST msg=audit({fake_timestamp_ns}): method={self.command} path={self.path} version={self.request_version} code={code} size={size}")
        # headers_str = ', '.join([f"{header}={value}" for header, value in self.headers.items()])
        # logging.info(f"type=HEADERS msg=audit({fake_timestamp_ns}): {headers_str} human_readable_timestamp={human_readable_timestamp}")

        # Write the request details to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", "fakesyscall", fake_timestamp_ns])

    def do_GET(self):
        print("received GET request", flush=True)
        # Handle GET request
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Brute Force Received!")
        self.log_request(200)

class RequestHandlerTicket(http.server.BaseHTTPRequestHandler):
    def __init__(self):
        super().__init__()
        csv_file = "ticket.csv"
        csv_headers = ['subjectname', 'subject_type', 'objectname', 'object_type', 'syscall', 'timestamp']

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)
    
    def log_request(self, syscall=""):
        fake_timestamp_ns = self.headers.get('X-Fake-Timestamp', None)
        if fake_timestamp_ns is None:
            fake_timestamp_ns = time.time_ns()
        else:
            fake_timestamp_ns = int(float(fake_timestamp_ns))        # Log the request details
            fake_timestamp_s = fake_timestamp_ns / 1e9
            human_readable_timestamp = datetime.fromtimestamp(fake_timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

            print(f"Received request at {human_readable_timestamp}", flush=True)

        # logging.info(f"type=REQUEST msg=audit({fake_timestamp_ns}): method={self.command} path={self.path} version={self.request_version} code={code} size={size}")
        # headers_str = ', '.join([f"{header}={value}" for header, value in self.headers.items()])
        # logging.info(f"type=HEADERS msg=audit({fake_timestamp_ns}): {headers_str} human_readable_timestamp={human_readable_timestamp}")

        # Write the request details to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["fake_subjectname", "fake_subjecttype", "fake_objectname", "fake_objecttype", "fakesyscall", fake_timestamp_ns])

    def do_GET(self):
        print("received GET request", flush=True)

        ticket_timestamp = self.headers.get('X-Ticket-Timestamp', None)
        curr_timestamp = self.headers.get('X-Fake-Timestamp', None)

        if ticket_timestamp is None or int(ticket_timestamp) < curr_timestamp:
            self.send_response(403)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            response = {"message": "Access Denied!"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            self.log_request("access_denied")
            return

        # Handle GET request
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"message": "Access Success!"}
        self.wfile.write(json.dumps(response).encode('utf-8'))
        self.log_request("access_success")

    # Let's use the POST to stimulate the ticketing system
    def do_POST(self):
        print("received POST request", flush=True)

        ticket_timestamp = self.headers.get('X-Ticket-Timestamp', None)
        curr_timestamp = self.headers.get('X-Fake-Timestamp', None)

        new_ticket_timestamp = curr_timestamp + 600 * 1e9
        # Handle POST request
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"ticket_timestamp": new_ticket_timestamp, "message": "Ticket Request Received!"}
        self.wfile.write(json.dumps(response).encode('utf-8'))
        self.log_request("ticket")

def run(server_class=http.server.HTTPServer, handler_class=RequestHandler, port=9500, csv_file=None):
    def handler(*args, **kwargs):
        handler_class(*args, csv_file=csv_file, **kwargs)
    
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}", flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="server configs")
    parser.add_argument("--server_type", type=str, help="Type of server run")
    parser.add_argument("--log_name", type=str, help="Name of the log file")

    args = parser.parse_args()

    if args.server_type == "dos":
        run(handler_class=RequestHandler, csv_file=args.log_name)
    elif args.server_type == "brute_force":
        run(handler_class=RequestHandlerBruteForce, csv_file=args.log_name)
    elif args.server_type == "ticket":
        run(handler_class=RequestHandlerTicket, csv_file=args.log_name)