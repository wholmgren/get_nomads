"""
A script to download weather forecast point data that is relevant to
solar power. Data is from the GFS model data hosted on the NOAA NOMADS
server. The script saves the data in a csv file.

The script supports simultaneous queries of different initialization times.

Run ``python get_nomads.py -h`` to see usage options.

Written by Will Holmgren, January 2017.
Copyright University of Arizona Board of Regents.
"""

import argparse
import os
from collections import defaultdict
import time
import logging

import pandas as pd
import numpy as np

from multiprocessing import Pool, TimeoutError

from pydap.client import open_url
from webob.exc import HTTPError

# some other variables of interest are commented out for efficiency
DEFAULT_VARIABLES = (
    'Temperature_surface',
#              'Wind_speed_gust',
     'U-component_of_wind',
     'V-component_of_wind',
     'Total_cloud_cover_entire_atmosphere',
#              'Total_cloud_cover_high_cloud',
#              'Total_cloud_cover_low_cloud',
#              'Total_cloud_cover_middle_cloud',
#              'Downward_Short-Wave_Rad_Flux'
    )


def multiple_times(init_timestamp, hours, lat_idx, lon_idx, variables):
    route = 'https://nomads.ncdc.noaa.gov/thredds/dodsC/gfs-004/'
    init_timestr = init_timestamp.strftime('%Y%m/%Y%m%d/gfs_4_%Y%m%d_%H%M')

    multiple_time_data = {}
    for fx_hour in range(3, hours+1, 3):
        full_uri = ('{route}{init_timestr}_{fx_hour:0>3}.grb2'.format(
            route=route, init_timestr=init_timestr, fx_hour=fx_hour))
        logging.info('getting data from %s', full_uri)
        dataset = open_url(full_uri)
        fx_timestamp = init_timestamp + pd.Timedelta(fx_hour, unit='h')
        try:
            single_time_data = single_time(dataset, lat_idx, lon_idx,
                                           variables)
        except HTTPError as e:
            logging.warning('error getting %s retrying in 30s.', full_uri)
            time.sleep(30)
            try:
                single_time_data = single_time(dataset, lat_idx, lon_idx,
                                               variables)
            except HTTPError as e:
                logging.warning(
                    '2nd error getting %s retrying in 30s.', full_uri)
                time.sleep(30)
                single_time_data = single_time(dataset, lat_idx, lon_idx,
                                               variables)
        except KeyError as e:
            logging.error('KeyError in %s', full_uri)

        multiple_time_data[fx_timestamp] = single_time_data

    return multiple_time_data


def single_time(dataset, lat_idx, lon_idx, variables):
    datad = {}
    for var in variables:
        datavar = dataset[var]
        data = get_datavar(datavar.array, var, lat_idx, lon_idx)
        datad[var] = np.asscalar(data)
    return datad


def get_datavar(datavar, var, lat_idx, lon_idx, is_retry=0):
    """Makes a http request for data and returns it.

    Calls itself up to 10 times if there's a network failure.
    """
    max_retry = 10
    sleep_time = 30

    if is_retry > max_retry:
        raise HTTPError('exceeded maximum retry attempts for %s', datavar)

    try:
        if 'wind' in var:
            data = datavar[0, 0, lat_idx, lon_idx]
        else:
            data = datavar[0, lat_idx, lon_idx]
    except (HTTPError, OSError) as e:
        logging.warning('error %s getting %s, retrying in %s s',
                        is_retry, datavar, sleep_time)
        time.sleep(sleep_time)
        data = get_datavar(datavar, var, lat_idx, lon_idx, is_retry=is_retry+1)

    return data


def construct_dataframe(multiple_time_data):
    index = pd.DatetimeIndex(sorted(multiple_time_data), tz='UTC')

    df = defaultdict(list)
    for k, v in sorted(multiple_time_data.items()):
        for var, data in v.items():
            df[var].append(data)
    df = pd.DataFrame(df, index=index)

    return df


def single_forecast(init_time, args):
    init_timestamp = pd.Timestamp(init_time)

    multiple_time_data = multiple_times(init_timestamp, args.hours,
                                        args.lat_idx, args.lon_idx,
                                        args.variables)

    df = construct_dataframe(multiple_time_data)

    output_file = init_timestamp.strftime('%Y%m%dT%H%M') + '.out' + '.csv'
    output_abspath = os.path.join(args.output_dir, output_file)
    df.to_csv(output_abspath)
    return output_abspath


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Download forecast data from NOMADS and export as csv')
    argparser.add_argument('-i', '--init-time',
                           help='initialization timestamp', action='append')
    argparser.add_argument('--hours', help='number of forecast hours',
                           default=24, type=int)
    argparser.add_argument('--init-time-start', help='first init-time')
    argparser.add_argument('--init-time-periods', help='number of init-times',
                           type=int)
    argparser.add_argument('--init-time-freq', help='frequency of init-times')
    argparser.add_argument('--lat-idx', help='latitude index', type=int)
    argparser.add_argument('--lon-idx', help='longitude index', type=int)
    argparser.add_argument('--output-dir', help='output file',
                           default=None)
    argparser.add_argument('-v', '--verbose',
                           help='Increase logging verbosity',
                           action='count')
    argparser.add_argument('--variables',
                           help='forecast variables (comma separated)',
                           default=DEFAULT_VARIABLES)
    argparser.add_argument('--timeout', help='request timeout', type=int,
                           default=None)
    argparser.add_argument('--processes', help='number of processes', type=int,
                           default=8)
    # pydap cache not working https://github.com/pydap/pydap/pull/37
    # argparser.add_argument('-c', '--cache', help='pydap cache', default=None)

    args = argparser.parse_args()

    if args.init_time_start is not None:
        args.init_time = pd.DatetimeIndex(start=args.init_time_start,
                                          freq=args.init_time_freq,
                                          periods=args.init_time_periods)

    args.output_dir = args.output_dir or os.getcwd()

    process_sets = np.ceil(len(args.init_time) / args.processes)
    args.timeout = args.timeout or int(args.hours * 60 / 3 * process_sets)

    # pydap.lib.CACHE = args.cache

    return args


def execute(init_times, args, p):
    multiple_results = {
        init_time: p.apply_async(single_forecast, args=(init_time, args))
        for init_time in init_times}

    successes = []
    failures = []
    for init_time, res in multiple_results.items():
        try:
            res.get(timeout=args.timeout)
        except (HTTPError, TimeoutError, OSError):
            logging.exception('failed to get forecast for %s', init_time)
            failures.append(init_time)
        else:
            logging.info('succeeded in getting forecast for %s', init_time)
            successes.append(init_time)

    logging.info('successes:\n%s', successes)
    logging.critical('failures:\n%s', failures)

    return failures


def main():
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')

    args = parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    with Pool(processes=args.processes) as p:
        failures = execute(args.init_time, args, p)
        if len(failures):
            logging.warning('retrying failed init times')
            failures = execute(failures, args, p)
            logging.critical('repeated failures:\n%s', failures)


if __name__ == '__main__':
    main()
