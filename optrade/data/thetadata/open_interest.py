# Copied (unedited) from ThetaData API documentation:
import httpx  # install via pip install httpx
import csv

BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

# set params
params = {
  'root': 'AAPL',
  'exp': '20250117',
  'right': 'C',
  'strike': '225000',
  'start_date': '20241107',
  'end_date': '20241107',
  'use_csv': 'true',
}
#
# This is the non-streaming version, and the entire response
# will be held in memory.
#
url = BASE_URL + '/hist/option/open_interest'

while url is not None:
    response = httpx.get(url, params=params, timeout=10)  # make the request
    response.raise_for_status()  # make sure the request worked

    # read the entire response, and parse it as CSV
    csv_reader = csv.reader(response.text.split("\n"))

    for row in csv_reader:
        print(row)  # do something with the data

    # check the Next-Page header to see if we have more data
    if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
        url = response.headers['Next-Page']
        params = None
    else:
        url = None


#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/hist/option/open_interest'

while url is not None:
    with httpx.stream("GET", url, params=params, timeout=10) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            print(line)  # do something with the data

    # check the Next-Page header to see if we have more data
    if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
        url = response.headers['Next-Page']
        params = None
    else:
        url = None
