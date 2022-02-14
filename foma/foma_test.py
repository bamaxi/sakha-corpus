import unittest
import csv
import subprocess
import logging
import time

logging.basicConfig(filename='foma_test.log', level=logging.DEBUG)

source = 'sakha_guess.foma'
BIN_FOMA = 'sakha_guess.bin'
START_LOG = 'foma_start.log'

TESTS_FILE = 'foma_test.csv'
RESULTS_FILE = 'results.csv'


def read_csv_dict(file):
    rows = []
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def read_stream(stream):
    lines = []
    break_next = False
    line = "%START%"
    while stream:
        if not line:
            break
        stream_readline = stream.readline()
        # logging.debug(f"readline is {stream_readline}")
        line = stream_readline.decode('utf-8').strip()
        logging.debug(f"line is: {line}")
        lines.append(line)
        if break_next:
            break
        if 'defined Grammar:' in line:
            break_next = True

    return "\n".join(lines)


def run_one_command(echo_cmd, flookup_cmd, wait=1):
    proc = subprocess.run(
        f"/bin/{' '.join(echo_cmd)} | {' '.join(flookup_cmd)}".encode('utf-8'),
        stdout=subprocess.PIPE,
        shell=True)  # TODO: shouldn't generally use shell=True
    return proc.stdout.decode('utf-8').strip('\n')


def capture_results(ignore_guess=True):
    results = []
    items = read_csv_dict(TESTS_FILE)
    logging.info("read test items")

    for item in items:
        logging.info(f"item is: {item}")
        flag = '-i' if item['query_type'] == 'down' else ''
        echo_cmd = ('echo', f'"{item["query"]}"')
        flookup_cmd = ('flookup', f'{flag}', '-b', '-x', f'{BIN_FOMA}')

        result = run_one_command(echo_cmd, flookup_cmd)
        logging.info(f"res is: {result}")

        item['actual'] = result
        item['expected'] = item['expected'].replace(' ', '\n')
        if ignore_guess:
            item['is_match'] = (item['actual'].strip('GUESS+')
                                == item['expected'].strip('GUESS+'))
        else:
            item['is_match'] = item['actual'] == item['expected']
        results.append(item)

    return results


if __name__ == '__main__':
    results = capture_results()

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
