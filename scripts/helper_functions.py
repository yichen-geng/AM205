import json
import os.path
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from obspy import Trace
from obspy import UTCDateTime
from obspy import read
from obspy.taup import TauPyModel
from scipy.signal import hilbert

sampling_rate = 10
half_win_len = 20
seconds_before_pick = 2.5
SIGNAL_WIN_LEN = 10
NOISE_WIN_LEN = 10
CORR_WIN_LEN = 10
# define PREM/IASP91 model
taup_model = TauPyModel("prem")


# # set random seed for bootstrapping
# np.random.seed(0)
# n_trials = 100  # number of bootstrapping trails


class Constants:
    R = 6371
    r_core_mantle = 3480
    r_inner_core = 1221.5
    max_distance = 40  # maximum distance of travel time
    epsilon = 1  # increment in distance


def preprocess(event_dir, origin_time, ev_lat, ev_lon, ev_dep, ev_mag, M, phase, data_dir, plot_dir,
               decimation=True, min_freq='raw', max_freq='raw', noise=False):
    """
    This function preprocesses traces for an event. It obtains distances, azimuths and takeoff angles for P and PcP waves
    for stations/traces available for cross-correlation.
    For the input phase, for each trace available, it calculates,
    1. the signal-to-noise ratio,
    2. a 2*half_win_len window centered at the predicted arrival time,
    3. a 10s centered at the maximum absolute amplitude of the phase (unless specified otherwise)
    4. the time shift from the predicted time to center the 10s window at the maximum absolute amplitude of the phase
    :param event_dir: str, directory at which the waveform data for the event are stored
    :param origin_time: datetime.datetime, the origin time of the event in UTC
    :param ev_lat: float, event latitude
    :param ev_lon: float, event longitude
    :param ev_dep: float, event depth
    :parame M: (3,3) array, moment tensor
    :param phase: str, 'P' or 'PcP'
    :param data_dir: str, directory at which the metadata will be stored
    :param plot_dir: str, directory at which the plots will be saved
    :param decimation: boolean, whether to decimate the trace, default: True
    :param min_freq: str or float, minimum corner frequency, default: 'raw'
    :param max_freq: str or float, maximum corner frequency, default: 'raw'
    :param noise: boolean, if True, noise data will be generated; otherwise, earthquake data will be preprocessed,
    default: False
    :return:
    station_names_sorted: array, (N,) array that contains the station names sorted by snr
    N is the number of stations/traces
    distances_sorted: array, (N,) array that contains the distances sorted by snr
    travel_times_predicted_sorted: array, (N,) array that contains the predicted travel times sorted by snr
    sn_ratios_sorted: array, (N,) array that contains descending signal-to-noise ratios
    predicted_windows_sorted: array, (N,2) array that contains the start and end indices of the predicted windows,
    sorted by snr
    final_windows_sorted: array, (N,2) array that contains the start and end indices of the final windows,
    sorted by snr
    total_time_shifts_sorted: array, (N,) array that contains the total time shifts from predicted times,
    sorted by snr
    stream_short_data_sorted: array, (N,M) array that contains the trace data, sorted by snr;
    M is the number of samples
    sampling_rate: int, new sampling rate
    # TODO: reorganize
    """
    stations = pd.read_csv("../../../data/Hinet/hinet_stations.csv")

    # calculate distances to Hinet stations
    st_lats = stations['latitude'].values
    st_lons = stations['longitude'].values
    distances, _, _ = calculate_distaz(st_lats, st_lons, ev_lat, ev_lon)

    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # load travel time table
    with open("../../../data/prem_model/travel_time_table_1km.json", 'r') as fp:
        tt_table = json.load(fp)

    # load crossover distance table
    with open("../../../data/prem_model/crossover_distance_table_1km.json", 'r') as fp:
        cd_table = json.load(fp)

    ev_dep_key = str(int(np.around(ev_dep)))  # round to the nearest integer
    tt_table = tt_table[ev_dep_key][phase]
    distance_table = np.array(tt_table['distances'])
    travel_time_table = np.array(tt_table['tts'])
    ray_param_table = np.array(tt_table['rps'])
    cd_table = cd_table[ev_dep_key]

    # calculate min and max time
    if phase == 'P':
        P_distances, P_travel_times, _ = get_travel_time('P', ev_dep, max_d=max_distance, first=True)
        # update the minimum distance to which P is observable
        if min_distance < min(P_distances):
            min_distance = min(P_distances)

    # TODO: allow for other phases
    # min_time and max_time are calculated for plotting only, so it is fine to not have the exact predictions
    flags_tmp = (distance_table >= min_distance) & (distance_table <= max_distance)
    distances_grid = distance_table[flags_tmp]
    travel_times_grid = travel_time_table[flags_tmp]

    # # calculate min and max time
    # distances_grid, travel_times_grid, _ = get_travel_time(phase, ev_dep, max_d=max_distance, first=True)

    # min_time and max_time are calculated for plotting only, so it is fine to not have the exact predictions
    min_time = travel_times_grid[0] - 50  # 50s before the min arrival time
    max_time = travel_times_grid[-1] + 50  # 50s after the max arrival time

    ev_time_JST = UTCDateTime(origin_time) + timedelta(hours=9)  # convert to Japan Standard Time

    # store metadata for all stations
    # (name, distance, azimuth, backazimuth, travel time, ray parameter, takeoff angle, radiation amplitude,
    # crossover phases, download status and waveform data completeness)
    station_metadata = {}
    stream_short_data = []
    stream_long_data = []
    station_names = []
    distances = []
    travel_times_predicted = []
    slownesses = []
    sn_ratios = []
    predicted_windows = []
    final_windows = []
    total_time_shifts = []
    crossover_phases = []  # store a list of crossover phases for the event
    n_stations_crossover = 0  # store the number of stations in the crossover distance range
    crossover_statuses = []

    global sampling_rate

    if decimation:
        sampling_rate = 10
    else:
        sampling_rate = 100

    # note that the actual number of stations processed is different
    print("Number of stations:", len(stations))

    for i in range(len(stations)):
        st_crossover_phases = []  # store a list of crossover phases for the station

        st = stations.iloc[i]
        st_name = st['name']

        # get station latitude and longitude
        st_lat = st['latitude']
        st_lon = st['longitude']

        # calculate the angular distance between the event and the station
        distance, azimuth, backazimuth = calculate_distaz(st_lat, st_lon, ev_lat, ev_lon)

        station_metadata[st_name] = {'latitude': st_lat, 'longitude': st_lon,
                                     'distance': distance, 'azimuth': azimuth, 'backazimuth': backazimuth}

        # check whether the phase can be observed at this distance
        # calculate predicted arrival time
        if noise:
            arrivals = taup_model.get_travel_times(source_depth_in_km=ev_dep, distance_in_degree=distance,
                                                   phase_list=['P'])
        else:
            arrivals = taup_model.get_travel_times(source_depth_in_km=ev_dep, distance_in_degree=distance,
                                                   phase_list=[phase])

        if len(arrivals) == 0:
            station_metadata[st_name] = {'travel_time': np.nan, 'ray_param': np.nan, 'takeoff_angle': np.nan,
                                         'radiation_amp': np.nan, 'crossover_phases': ()}
        else:
            travel_time = arrivals[0].time
            ray_param = arrivals[0].ray_param

            # # calculate the incidence angle at the source
            # takeoff_angle = calculate_p_incidence_angle_from_radial_distance(ray_param, Constants.R - ev_dep)
            #
            # # calculate the radiation amplitude
            # radiation_amp = calculate_p_wave_radiation_amplitude(M, takeoff_angle, azimuth)

            # check whether the distance falls within the crossover distance range
            crossover = False
            for key, _ in cd_table.items():
                if key == "P" or key == "S":
                    # for P or S, consider all arrivals within 10s
                    # for P, "all" or "first" is the same because it crosses with PcP at large distances
                    # (e.g., > 80 degrees), whereas multiple P arrivals (triplication) are observed at small distances
                    # (e.g., < 30 degrees)
                    cd_ranges = cd_table[key]["10"]["all"]
                else:
                    # for other phases, consider the first arrival within 5s
                    cd_ranges = cd_table[key]["5"]["first"]
                for cd_range in cd_ranges:
                    if len(cd_range) != 0:
                        min_cd = cd_range[0]
                        max_cd = cd_range[1]
                        if min_cd <= distance <= max_cd:
                            # print(distance, key, cd_range)
                            crossover = True
                            # save
                            st_crossover_phases.append(key)
                            if key not in crossover_phases:
                                crossover_phases.append(key)
                            break

            # # TODO: modify on 12/3/22, only remove sP crossover
            # for key, _ in cd_table.items():
            #     if key == "sP":
            #         # for other phases, consider the first arrival within 5s
            #         cd_ranges = cd_table[key]["5"]["first"]
            #         for cd_range in cd_ranges:
            #             if len(cd_range) != 0:
            #                 min_cd = cd_range[0]
            #                 max_cd = cd_range[1]
            #                 if min_cd <= distance <= max_cd:
            #                     # print(distance, key, cd_range)
            #                     crossover = True
            #                     # save
            #                     st_crossover_phases.append(key)
            #                     if key not in crossover_phases:
            #                         crossover_phases.append(key)
            #                     break

            if crossover:
                n_stations_crossover += 1

            station_metadata[st_name]['travel_time'] = travel_time
            station_metadata[st_name]['ray_param'] = ray_param
            # station_metadata[st_name]['takeoff_angle'] = takeoff_angle
            # station_metadata[st_name]['radiation_amp'] = radiation_amp
            station_metadata[st_name]['crossover_phases'] = tuple(st_crossover_phases)

            # TODO: SAC files are velocity in nm/s or accelaration in nm/s/s
            file_dir = event_dir + "/" + st_name + ".U.SAC"
            download_status = os.path.isfile(file_dir)
            station_metadata[st_name]['download'] = int(download_status)

            # proceed if the waveform data exist
            if download_status:
                if noise:
                    trace_short_data, trace_long_data, data_status = \
                        preprocess_single_trace(file_dir, decimation, min_freq, max_freq, ev_time_JST, travel_time,
                                                dt1=-210, dt2=-10, min_time=min_time, max_time=max_time)
                else:
                    trace_short_data, trace_long_data, data_status = \
                        preprocess_single_trace(file_dir, decimation, min_freq, max_freq, ev_time_JST, travel_time,
                                                min_time=min_time, max_time=max_time)

                station_metadata[st_name]['data'] = int(data_status)

                # proceed if the waveform data have the correct sampling rate and record length and
                # the station is not in the crossover distance range
                if data_status:
                    if noise:
                        # generate a random number between -20 and 20
                        delta_t = random.randint(-20, 20)
                    else:
                        delta_t = 0

                    # calculate signal-to-noise ratio
                    sn_ratio = calculate_signal_to_noise_ratio(trace_short_data, sampling_rate, noise_window_len=10,
                                                               noise=noise, delta_t=delta_t)

                    # select a half_win_len*2 s window centered at the predicted arrival time
                    _, predicted_start, predicted_end = \
                        select_trace_window(trace_short_data, sampling_rate,
                                            init_window_len=half_win_len * 2, selected_window_len=half_win_len * 2,
                                            predicted=True, noise=noise, delta_t=delta_t)

                    # select a 10 s window centered at the maximum absolute amplitude
                    total_time_shift, selected_start, selected_end = \
                        select_trace_window(trace_short_data, sampling_rate, noise=noise, delta_t=delta_t)

                    # TODO: do not keep trace/stream object, convert to numpy array instead

                    stream_short_data.append(trace_short_data)
                    stream_long_data.append(trace_long_data)
                    station_names.append(st_name[-4:])
                    distances.append(distance)
                    travel_times_predicted.append(travel_time)
                    slownesses.append(ray_param * np.pi / 180)  # convert to s/degree
                    sn_ratios.append(sn_ratio)
                    predicted_windows.append([predicted_start, predicted_end])
                    final_windows.append([selected_start, selected_end])
                    total_time_shifts.append(float(total_time_shift))
                    crossover_statuses.append(crossover)
            else:
                station_metadata[st_name]['data'] = 0

    # convert to numpy.array
    stream_short_data = np.array(stream_short_data)
    stream_long_data = np.array(stream_long_data)
    station_names = np.array(station_names)
    distances = np.array(distances)
    travel_times_predicted = np.array(travel_times_predicted)
    slownesses = np.array(slownesses)
    sn_ratios = np.array(sn_ratios)
    predicted_windows = np.array(predicted_windows)
    final_windows = np.array(final_windows)
    total_time_shifts = np.array(total_time_shifts)
    crossover_statuses = np.array(crossover_statuses)

    print("Number of crossover stations:", n_stations_crossover)
    print("Crossover phases:", crossover_phases)

    # save station metadata
    save_dir = data_dir + "/" + origin_time.strftime("%Y%m%d%H%M%S%f") + "_station_metadata.csv"
    # if not os.path.isfile(save_dir):
    df_stations = pd.DataFrame(station_metadata).T
    df_stations.to_csv(save_dir, index_label='station')

    # check the polarity
    radiation_amps = df_stations['radiation_amp'].values
    polarity_status = int((sum(radiation_amps > 0) == len(radiation_amps)) or
                          (sum(radiation_amps < 0) == len(radiation_amps)))

    # order traces by descending signal-to-noise ratios
    sort_indices = np.flip(np.argsort(sn_ratios))
    n_traces = len(sort_indices)

    stream_short_data_sorted = stream_short_data[sort_indices]
    stream_long_data_sorted = stream_long_data[sort_indices]
    station_names_sorted = station_names[sort_indices]
    distances_sorted = distances[sort_indices]
    travel_times_predicted_sorted = travel_times_predicted[sort_indices]
    slownesses_sorted = slownesses[sort_indices]
    sn_ratios_sorted = sn_ratios[sort_indices]
    predicted_windows_sorted = predicted_windows[sort_indices]
    final_windows_sorted = final_windows[sort_indices]
    total_time_shifts_sorted = total_time_shifts[sort_indices]
    crossover_statuses_sorted = crossover_statuses[sort_indices]

    # test array lengths are correct
    assert len(stream_short_data_sorted) == n_traces
    assert len(stream_long_data_sorted) == n_traces
    assert len(station_names_sorted) == n_traces
    assert len(distances_sorted) == n_traces
    assert len(travel_times_predicted_sorted) == n_traces
    assert len(slownesses_sorted) == n_traces
    assert len(sn_ratios_sorted) == n_traces
    assert len(predicted_windows_sorted) == n_traces
    assert len(final_windows_sorted) == n_traces
    assert len(total_time_shifts_sorted) == n_traces
    assert len(crossover_statuses_sorted) == n_traces

    # plot
    # with crossover distances
    # select good traces for plotting
    _, _, indices_selected = select_traces(distances_sorted, sn_ratios_sorted)

    title = str(origin_time) + ", " + str(ev_lat) + ", " + str(ev_lon) + ", " + str(ev_dep) + "km, Mw" + str(ev_mag)

    # # plot long record section with travel times
    # fig1, ax1 = plt.subplots(1, 1, figsize=(18, 8))
    # plot_record_section_long(stream_long_data_sorted[indices_selected], distances_sorted[indices_selected],
    #                          min_time, max_time, ax1, title=title)
    # plt.savefig(plot_dir + "/record_section_long_with_cd.png", dpi=150)
    # # # save a copy of the record section to the folder shared by all events
    # # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
    # #             "_record_section_long_with_cd.png", dpi=150)
    # # superimpose travel times
    # plot_travel_times(ev_dep, max(distances_sorted), ax1)
    # plt.legend()
    # plt.savefig(plot_dir + "/record_section_long_travel_times_with_cd.png", dpi=150)
    # # # save a copy of the record section to the folder shared by all events
    # # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
    # #             "_record_section_long_travel_times_with_cd.png", dpi=150)
    # plt.clf()
    # plt.close(fig1)

    # # plot short record section
    # fig2, ax2 = plt.subplots(1, 1, figsize=(15, 3))
    # plot_record_section_short(stream_short_data_sorted[indices_selected], distances_sorted[indices_selected],
    #                           predicted_windows_sorted[indices_selected], ax2, title=title)
    # plt.savefig(plot_dir + "/record_section_short_with_cd.png", dpi=150)
    # # # save a copy of the record section to the folder shared by all events
    # # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
    # #             "_record_section_short_with_cd.png", dpi=150)
    # plt.clf()
    # plt.close(fig2)

    if sum(~crossover_statuses_sorted) > 1:
        # without crossover distances
        distances_plot = distances_sorted[~crossover_statuses_sorted]
        sn_ratios_plot = sn_ratios_sorted[~crossover_statuses_sorted]
        stream_short_data_plot = stream_short_data_sorted[~crossover_statuses_sorted]
        # stream_long_data_plot = stream_long_data_sorted[~crossover_statuses_sorted]
        predicted_windows_plot = predicted_windows_sorted[~crossover_statuses_sorted]

        # select good traces for plotting
        _, _, indices_selected = select_traces(distances_plot, sn_ratios_plot)

        title = str(origin_time) + ", " + str(ev_lat) + ", " + str(ev_lon) + ", " + str(ev_dep) + "km, Mw" + str(ev_mag)

        # # plot long record section with travel times
        # fig1, ax1 = plt.subplots(1, 1, figsize=(18, 8))
        # plot_record_section_long(stream_long_data_plot[indices_selected], distances_plot[indices_selected],
        #                          min_time, max_time, ax1, title=title)
        # plt.savefig(plot_dir + "/record_section_long_wo_cd.png", dpi=150)
        # # save a copy of the record section to the folder shared by all events
        # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
        #             "_record_section_long_wo_cd.png", dpi=150)
        # # superimpose travel times
        # plot_travel_times(ev_dep, max(distances_plot), ax1)
        # plt.legend()
        # plt.savefig(plot_dir + "/record_section_long_travel_times_wo_cd.png", dpi=150)
        # # save a copy of the record section to the folder shared by all events
        # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
        #             "_record_section_long_travel_times_wo_cd.png",
        #             dpi=150)
        # plt.clf()
        # plt.close(fig1)

        # # plot short record section
        # fig2, ax2 = plt.subplots(1, 1, figsize=(15, 3))
        # plot_record_section_short(stream_short_data_plot[indices_selected], distances_plot[indices_selected],
        #                           predicted_windows_plot[indices_selected], ax2, title=title)
        # plt.savefig(plot_dir + "/record_section_short_wo_cd.png", dpi=150)
        # # # save a copy of the record section to the folder shared by all events
        # # plt.savefig("../../plots/results/" + origin_time.strftime("%Y%m%d%H%M%S%f") +
        # #             "_record_section_short_wo_cd.png", dpi=150)
        # plt.clf()
        # plt.close(fig2)

    # include crossover distances
    station_names_final = station_names_sorted
    stream_data_final = stream_short_data_sorted
    distances_final = distances_sorted
    travel_times_predicted_final = travel_times_predicted_sorted
    slownesses_final = slownesses_sorted
    predicted_windows_final = predicted_windows_sorted
    final_windows_final = final_windows_sorted
    total_time_shifts_final = total_time_shifts_sorted
    sn_ratios_final = sn_ratios_sorted
    crossover_statuses_final = crossover_statuses_sorted

    # # exclude crossover distances
    # station_names_final = station_names_sorted[~crossover_statuses_sorted]
    # stream_data_final = stream_short_data_sorted[~crossover_statuses_sorted]
    # distances_final = distances_sorted[~crossover_statuses_sorted]
    # travel_times_predicted_final = travel_times_predicted_sorted[~crossover_statuses_sorted]
    # slownesses_final = slownesses_sorted[~crossover_statuses_sorted]
    # predicted_windows_final = predicted_windows_sorted[~crossover_statuses_sorted]
    # final_windows_final = final_windows_sorted[~crossover_statuses_sorted]
    # total_time_shifts_final = total_time_shifts_sorted[~crossover_statuses_sorted]
    # sn_ratios_final = sn_ratios_sorted[~crossover_statuses_sorted]

    return station_names_final, stream_data_final, distances_final, travel_times_predicted_final, slownesses_final, \
           predicted_windows_final, final_windows_final, total_time_shifts_final, sn_ratios_final, crossover_statuses_final, \
           distances_grid, travel_times_grid, polarity_status


def preprocess_single_trace(file_dir, decimation, min_freq, max_freq, ev_time_JST, travel_time, dt1=-100., dt2=100.,
                            min_time=None, max_time=None):
    """
    This function preprocesses a single trace (vertical component).
    :param file_dir: str, directory at which the trace is stored
    :param decimation: boolean, whether to decimate the trace, default: True
    :param min_freq: str or float, minimum corner frequency, default: 'raw'
    :param max_freq: str or float, maximum corner frequency, default: 'raw'
    :param ev_time_JST: obspy.core.UTCDateTime, event time in JST
    :param travel_time: float, travel time of the phase in seconds
    :param dt1: float, trace start time with respect to the arrival time in seconds
    :param dt2: float, trace end time with respect to the arrival time in seconds
    :param min_time: # TODO
    :param max_time: # TODO
    :return:
    trace_BHZ: obspy.core.trace.Trace, 200s normalized trace centered at the predicted arrival time
    data_status: boolean, True if preprocessing succeeds, False otherwise
    """
    # initialize
    # trace_short = Trace()
    trace_long = Trace()
    data_status = False

    # 1. read trace
    stream = read(file_dir)
    trace_BHZ = stream[0]

    sr_old = trace_BHZ.stats.sampling_rate

    # some traces have sampling rate < 100 Hz or total length < 25 mins or a flat response
    if (sr_old == 100) and (len(trace_BHZ.data) == 25 * 60 * sr_old):
        # 2. (optional) decimate by a factor of 10
        if decimation:
            trace_BHZ.decimate(factor=10, no_filter=True)
            # sanity check
            # the original sampling rate of Hinet is 100 Hz, after decimating by a factor of 10,
            # the sampling rate should be 10 Hz
            sr = int(trace_BHZ.stats.sampling_rate)  # cast to int
            assert sr == 10
        else:
            # sanity check
            # the original sampling rate of Hinet is 100 Hz
            sr = int(trace_BHZ.stats.sampling_rate)  # cast to int
            assert sr == 100

        # 3. (optional) filter
        if min_freq != 'raw' and max_freq != 'raw':
            # apply band-pass filter
            # TODO: ask about zerophase
            trace_BHZ.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=2)
        elif min_freq != 'raw':
            # apply high-pass filter
            trace_BHZ.filter('highpass', freq=min_freq, corners=2)

        # trace_short = trace_BHZ.copy()
        trace_long = trace_BHZ.copy()

        # 4. select between dt1 and dt2 after the phase arrival for cross-correlation
        arrival_time = ev_time_JST + travel_time
        # # add +-10s to the arrival time for testing
        # arrival_time -= 10  # "late arrivals"
        # arrival_time += 10  # "early arrivals"
        trace_BHZ.trim(starttime=arrival_time + dt1, endtime=arrival_time + dt2)

        # check whether the trace has a flat response
        trace_data_tmp = trace_BHZ.data
        flat = np.all(trace_data_tmp == trace_data_tmp[0])

        if not flat:
            # 5. detrend
            trace_BHZ.detrend(type='linear')

            # 6. normalize for cross-correlation
            trace_BHZ.normalize()

            if (min_time is not None) and (max_time is not None):
                # 4. select between min_time and max_time after the event time for plotting
                trace_long.trim(starttime=ev_time_JST + min_time, endtime=ev_time_JST + max_time)
                # 5. detrend and demean
                trace_long.detrend(type='linear')
                # 6. normalize by the maximum absolute amplitude in a short time window around the phase arrival
                trace_tmp = trace_long.copy()
                trace_tmp.trim(starttime=arrival_time - half_win_len, endtime=arrival_time + half_win_len)
                trace_long.data = trace_long.data / max(abs(trace_tmp.data))

            data_status = True

    return trace_BHZ.data, trace_long.data, data_status


def calculate_distaz(st_lat, st_lon, ev_lat, ev_lon):
    """
    This function calculates the distance, azimuth and backazimuth for an event-station pair.
    :param st_lat: float, station latitude in degrees
    :param st_lon: float, station longitude in degrees
    :param ev_lat: float, event latitude in degrees
    :param ev_lon: float, event longitude in degrees
    :return:
    delta: float, distance in degrees
    azi: float, azimuth in radians
    baz: float, backazimuth in radians
    """
    # convert degree to radian
    st_lat = st_lat / 180 * np.pi
    st_lon = st_lon / 180 * np.pi

    ev_lat = ev_lat / 180 * np.pi
    ev_lon = ev_lon / 180 * np.pi

    phi_s = st_lat
    lambda_s = st_lon

    phi_e = ev_lat
    lambda_e = ev_lon

    d_lambda = abs(lambda_s - lambda_e)

    # great-circle distance
    delta = np.arccos(np.sin(phi_s) * np.sin(phi_e) + np.cos(phi_s) * np.cos(phi_e) * np.cos(d_lambda))

    # azimuth [-pi, pi]
    azi = np.arctan2(np.cos(phi_s) * np.cos(phi_e) * np.sin(lambda_s - lambda_e),
                     np.sin(phi_s) - np.cos(delta) * np.sin(phi_e))

    # convert to [0, 2*pi]
    azi = np.where(azi > 0, azi, azi + 2 * np.pi)

    baz = np.arctan2(-np.cos(phi_s) * np.cos(phi_e) * np.sin(lambda_s - lambda_e),
                     np.sin(phi_e) - np.cos(delta) * np.sin(phi_s))
    # convert to [0, 2*pi]
    baz = np.where(baz > 0, baz, baz + 2 * np.pi)

    # convert back to degree
    delta = delta / np.pi * 180

    return delta, azi, baz


def calculate_signal_to_noise_ratio(trace_data, sampling_rate,
                                    init_window_len=10, signal_window_len=4, noise_window_len=10, noise=False,
                                    delta_t=0):
    """
    This function calculates the signal-to-noise ratio for a phase.
    It first finds a signal window centered at the maximum absolute amplitude and then
    takes a noise window either before the P arrival or the signal window.
    Signal-to-noise ratio is calculated as the ratio between the mean absolute amplitude of two windows.
    :param trace_data: obspy.core.trace.Trace, raw trace
    :param phase: str, seismic phase
    :param sampling_rate: int, sampling rate of the raw trace
    :param taup_model: obspy.taup.TauPyModel, taup model to calculate the predicted travel time for the phase
    :param ev_time: obspy.core.utcdatetime.UTCDateTime, event time
    :param ev_depth: float, event depth in km
    :param distance: float, angular distance between the event and the station in degrees
    :param init_window_len: int, length of the initial window
    :param signal_window_len: int, length of the signal window
    :param noise_window_len: int, length of the noise window
    :param noise: boolean, if True, the predicted arrival time is randomized; otherwise, the predicted arrival time is
    calculated by TauP, default: False
    :param delta_t: int, the time shift from the predicted arrival time in seconds, used only when noise=True,
    default: 0
    :return: sn_ratio: float, signal-to-noise ratio of the phase
    """
    # _s signal window
    _, idx_start, idx_end = select_trace_window(trace_data, sampling_rate, init_window_len, signal_window_len,
                                                noise=noise, delta_t=delta_t)
    # get the signal window
    signal_data = trace_data[idx_start:idx_end]
    signal_data = detrend(signal_data)
    # calculate the mean amplitude
    signal_amplitude = np.mean(abs(signal_data))
    # get the noise window
    noise_data = trace_data[idx_start - int(noise_window_len * sampling_rate):idx_start]
    noise_data = detrend(noise_data)
    # calculate the mean amplitude
    noise_amplitude = np.mean(abs(noise_data))

    # calculate the signal-to-noise ratio
    sn_ratio = signal_amplitude / noise_amplitude

    return sn_ratio


def select_trace_window(trace_data, sampling_rate, init_window_len=10, selected_window_len=10, predicted=False,
                        noise=False, delta_t=0):
    """
    Select a signal window centered at the maximum absolute amplitude or at the predicted arrival time.
    :param trace_data: array, raw trace
    :param sampling_rate: int, sampling rate of the raw trace
    :param init_window_len: int, length of the initial window
    :param selected_window_len: int, length of the final window
    :param predicted: boolean, whether to center the trace at the predicted arrival time
    :param noise: boolean, if True, the predicted arrival time is randomized; otherwise, the predicted arrival time is
    calculated by TauP, default: False
    :param delta_t: int, the time shift from the predicted arrival time in seconds, used only when noise=True,
    default: 0
    :return:
    time_shift: int, difference in indices between max abs amplitude and predicted arrival
    start: int, index of the start of window
    end: int, index of the end of window
    """
    trace_len = len(trace_data)
    # trace is centered at the predicted arrival time, so idx_pred should be the midpoint
    idx_pred = int(trace_len / 2)  # midpoint

    if noise:
        # pick a point within +-20s of the midpoint as the PcP arrival time
        idx_pred += delta_t * sampling_rate

    # calculate start and end indices in the raw trace
    idx_start = idx_pred - int(np.around(
        init_window_len / 2 * sampling_rate))  # TODO: double check whether np.around() is needed here and same for the rest
    idx_end = idx_pred + int(np.around(init_window_len / 2 * sampling_rate))
    if predicted:
        return 0, idx_start, idx_end
    else:
        trace_window = trace_data[idx_start:idx_end]
        # find the index of the maximum absolute amplitude in the trace window
        # note that this index is defined with respect to the window not the full trace
        idx_max_rel = np.argmax(abs(trace_window))
        # calculate the difference between index of max abs amplitude (actual?) and index of predicted PcP arrival time
        time_shift = idx_max_rel - int(np.around(init_window_len / 2 * sampling_rate))
        # calculate the index of the maximum absolute amplitude with respect to the full trace
        idx_max = time_shift + idx_pred
        # center the trace window at the index of max abs amplitude
        idx_start = idx_max - int(np.around(selected_window_len / 2 * sampling_rate))
        idx_end = idx_max + int(np.around(selected_window_len / 2 * sampling_rate))
        return time_shift, idx_start, idx_end


def select_cross_correlation_window(trace_data, sampling_rate, pick=None, win_len=CORR_WIN_LEN, envelope=True):
    """
    This function selects a window around the pick time for cross-correlation. The time before the pick time and the
    window length are specified by the inputs. It also calculates the signal-to-noise ratio based on the pick time.
    :param trace_data: array, (M,) array that contains the trace data; M is the number of samples
    :param sampling_rate: int, sampling rate of the trace data
    :param pick: float, pick time in seconds
    :param win_len: int, length of the window in seconds
    :return:
    trace_window: array, (2,) array that contains the start and end indices of the selected window for the trace
    pick_new: float, new pick time for the trace in seconds
    """
    half_win_len_tmp = int(len(trace_data) / 2)

    # search for the maximum abs amplitude within +-20 s
    idx_search_start = int(np.around(half_win_len_tmp - 20 * sampling_rate))
    idx_search_end = int(np.around(half_win_len_tmp + 20 * sampling_rate))

    if envelope:
        # calculate the envelope
        analytic_signal = hilbert(trace_data)
        envelope = np.abs(analytic_signal)

        # # normalize
        # envelope /= max(envelope[idx_search_start:idx_search_end + 1])

        # find max
        idx_max = idx_search_start + np.argmax(envelope[idx_search_start:idx_search_end + 1])
        # # find the first high peak  # TODO: threshold?
        # idx_max = np.where(envelope > 0.8)[0][0]
    else:
        # find max
        idx_max = idx_search_start + np.argmax(abs(trace_data[idx_search_start:idx_search_end + 1]))

    # define a _s window centered at the maximum amplitude
    idx_start = idx_max - int(np.around(win_len / 2 * sampling_rate))
    idx_end = idx_max + int(np.around(win_len / 2 * sampling_rate))
    trace_window = np.array([idx_start, idx_end])

    if pick is not None:
        # recalculate the pick time
        # calculate the index of the pick time
        idx_pick = pick * sampling_rate + half_win_len_tmp
        pick_new = -5 + (idx_pick - idx_start) / sampling_rate

        return trace_window, pick_new
    else:
        return trace_window


def cross_correlate(stream_data, sampling_rate, init_windows, ref_trace_data=None, ref_trace_window=None,
                    lmax_shift=5, rmax_shift=5, flip_polarity=True, weights=None):
    """
    This function cross-correlates the traces with one another or with a reference trace.
    :param stream_data: array, (N,M) array that contains the trace data;
    N is the number of traces, and M is the number of samples
    :param sampling_rate: int, sampling rate of the trace data
    :param init_windows: array, (N,2) array that contains the start and end indices of the initial cross-correlation
    window for each trace
    :param ref_trace_data: array, (M,) array that contains the reference trace data, default: None
    :param ref_trace_window: array, (2,) array that contains the start and end indices of the cross-correlation window
    for the reference trace, default: None
    :param lmax_shift: int, maximum shift to the left in seconds, default: 5
    :param rmax_shift: int, maximum shift to the right in seconds, default: 5
    :param flip_polarity: boolean, if True, cross-correlation finds the maximum absolute correlation; otherwise,
    it finds the maximum (positive) correlation, default: True
    :param weights: # TODO
    :return:
    corr_vals: array, correlation values
    time_shifts: array, time shifts
    """
    # define maximum left and right shift
    lmax_shift *= sampling_rate
    rmax_shift *= sampling_rate

    # cast to integer
    lmax_shift = int(np.around(lmax_shift))
    rmax_shift = int(np.around(rmax_shift))

    n_traces = len(stream_data)

    if (ref_trace_data is not None) and (ref_trace_window is not None):
        # initialize empty arrays
        corr_vals = np.zeros(shape=(n_traces,))
        time_shifts = np.zeros(shape=(n_traces,))

        start1 = ref_trace_window[0]
        end1 = ref_trace_window[1]
        trace1_window = ref_trace_data[int(np.around(start1)):int(np.around(end1 + 1))]
        trace1_window = detrend(trace1_window)

        for j in np.arange(n_traces):
            # cross-correlate with another trace
            trace2_data = stream_data[j]
            start2 = init_windows[j][0]
            end2 = init_windows[j][1]

            xcorr = [None] * ((lmax_shift + rmax_shift) + 1)

            for k, shift in enumerate(range(-lmax_shift, rmax_shift + 1, 1)):
                # get the window for cross-correlation
                trace2_window = trace2_data[int(np.around(start2 + shift)):int(np.around(end2 + shift + 1))]
                trace2_window = detrend(trace2_window)

                # calculate the pearson correlation
                xcorr[k] = calculate_pearson_correlation(trace1_window, trace2_window)
            xcorr = np.array(xcorr)

            if flip_polarity:
                idx = np.argmax(abs(xcorr))
            else:
                idx = np.argmax(xcorr)  # positive correlation

            # get the maximum correlation value
            max_corr = xcorr[idx]
            # get the time shift
            optimal_shift = idx - lmax_shift

            # save to an array
            corr_vals[j] = max_corr
            time_shifts[j] = optimal_shift
    else:
        # initialize empty matrices
        corr_vals = np.zeros(shape=(n_traces, n_traces))
        time_shifts = np.zeros(shape=(n_traces, n_traces))

        for i in np.arange(n_traces):
            print(str(i + 1) + "/" + str(n_traces))
            trace1_data = stream_data[i]
            start1 = init_windows[i][0]
            end1 = init_windows[i][1]
            trace1_window = trace1_data[int(np.around(start1)):int(np.around(end1 + 1))]
            trace1_window = detrend(trace1_window)

            for j in np.arange(i, n_traces, 1):  # start from i-th station to avoid duplicated calculation
                # for j in np.arange(n_traces):
                # cross-correlate with another trace
                trace2_data = stream_data[j]
                start2 = init_windows[j][0]
                end2 = init_windows[j][1]

                xcorr = [None] * ((lmax_shift + rmax_shift) + 1)

                for k, shift in enumerate(range(-lmax_shift, rmax_shift + 1, 1)):
                    # get the window for cross-correlation
                    trace2_window = trace2_data[int(np.around(start2 + shift)):int(np.around(end2 + shift + 1))]
                    trace2_window = detrend(trace2_window)

                    # calculate the pearson correlation
                    xcorr[k] = calculate_pearson_correlation(trace1_window, trace2_window)
                xcorr = np.array(xcorr)

                if flip_polarity:
                    idx = np.argmax(abs(xcorr))
                else:
                    idx = np.argmax(xcorr)  # positive correlation

                # get the maximum correlation value
                max_corr = xcorr[idx]
                # get the time shift
                optimal_shift = idx - lmax_shift

                # save to a 2D matrix
                corr_vals[i][j] = max_corr
                time_shifts[i][j] = optimal_shift
        print("")

    return corr_vals, time_shifts


def detrend(y_data):
    """
    # TODO: least-square fit
    :param y_data:
    :return:
    """
    x_data = np.arange(len(y_data))
    beta0, beta1 = least_square_fit(x_data, y_data)
    y_data_detrend = y_data - (beta0 + beta1 * x_data)
    # # matrix form
    # X = x_data[:, np.newaxis]**[0, 1]
    # Y = y_data[:, np.newaxis]
    # p = scipy.linalg.inv(X.T @ X) @ X.T @ Y
    # y_data_detrend = y_data - (p[0] + p[1] * x_data)
    return y_data_detrend


def least_square_fit(x_data, y_data, weights=None):
    """

    :param x_data:
    :param y_data:
    :param weights:
    :return:
    """
    # non-matrix form
    if weights is not None:
        x_mean = np.sum(x_data * weights) / np.sum(weights)
        y_mean = np.sum(y_data * weights) / np.sum(weights)
        beta1 = np.sum((x_data - x_mean) * (y_data - y_mean) * weights) / np.sum((x_data - x_mean) ** 2 * weights)
    else:
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)
        beta1 = np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2)
    beta0 = y_mean - beta1 * x_mean
    return beta0, beta1


def calculate_pearson_correlation(data1, data2):
    """
    This function calculates the Pearson correlation coefficient between data1 and data2, assuming that trend and mean
    are removed.
    :param data1: array, data1
    :param data2: array, data2
    :return: r: float, the Pearson correlation coefficient.
    """
    top = np.sum(data1 * data2)
    bottom = np.sqrt(np.sum(data1 ** 2)) * np.sqrt(np.sum(data2 ** 2))
    r = top / bottom

    return r


def select_window(trace_data, sampling_rate, pick, time_before_pick=2.5, win_len=10):
    half_win_len_tmp = int(len(trace_data) / 2) / sampling_rate
    start = int(np.around((half_win_len_tmp + pick - time_before_pick) * sampling_rate))
    end = start + int(win_len * sampling_rate)
    trace_window = np.array([start, end])
    # the 10s window spans from -5s to 5s, so the new pick time (relative to 0) is
    # -5s + time_before_pick
    # e.g., if time_before_pick = 2.5s, then pick_new = -5s + time_before_pick = -5s + 2.5s = -2.5s
    pick_new = -5 + time_before_pick

    return trace_window, pick_new


def calculate_snr_by_pick(trace_data, sampling_rate, pick,
                          signal_window_len=SIGNAL_WIN_LEN, noise_window_len=NOISE_WIN_LEN,
                          signal_weights=None, noise_weights=None, method=None):
    """
    This function calculates the signal-to-noise ratio (snr) for a phase, assuming that the pick time is provided.
    The signal window starts from the pick time and the noise window ends before the pick time. The length of the
    signal/noise window is specified by the input.
    Three methods can be used to calculate the snr:
    1. ratio of the mean absolute amplitude between the signal and noise window (non-weighted)
    2. ratio of the root-mean-square between the signal and noise window (non-weighted)
    3. ratio of the mean absolute amplitude between the signal and noise window weighted by the values provided
    (weighted)
    :param trace_data: array, (M,) array that contains the trace data
    :param sampling_rate: int, sampling rate of the trace data
    :param pick: float, pick time in seconds
    :param signal_window_len: float, length of the signal window, default: SIGNAL_WIN_LEN
    :param noise_window_len: float, length of the noise window, default: NOISE_WIN_LEN
    :param signal_weights: array, (signal_window_len * sampling_rate,) array that contains the weights for the signal;
    if no weights are provided, the snr is calculated using non-weighted methods, default: None
    :param noise_weights: array, (noise_window_len * sampling_rate,) array that contains the weights for the noise;
    if no weights are provided, the snr is calculated using non-weighted methods, default: None
    :param method: str, method used to calculate the non-weighted snr;
    if the method='rms', method 2 will be used; otherwise, method 1 will be used, default: None
    :return:
    snr: float, the snr of the phase
    """
    half_win_len_tmp = int(len(trace_data) / 2) / sampling_rate
    # signal window
    signal_start = int(np.around((half_win_len_tmp + pick) * sampling_rate))
    signal_end = signal_start + int(signal_window_len * sampling_rate)
    signal_data = trace_data[signal_start:signal_end + 1]
    signal_data = detrend(signal_data)
    if signal_weights is not None:
        signal_amplitude = np.sum(abs(signal_data) * signal_weights) / np.sum(signal_weights)
    else:
        if method == 'rms':
            signal_amplitude = np.sqrt(np.mean(signal_data ** 2))
        else:
            signal_amplitude = np.mean(abs(signal_data))
    # noise window
    noise_data = trace_data[signal_start - int(noise_window_len * sampling_rate):signal_start]
    noise_data = detrend(noise_data)
    if noise_weights is not None:
        noise_amplitude = np.sum(abs(noise_data) * noise_weights) / np.sum(noise_weights)
    else:
        if method == 'rms':
            noise_amplitude = np.sqrt(np.mean(noise_data ** 2))
        else:
            noise_amplitude = np.mean(abs(noise_data))
    snr = signal_amplitude / noise_amplitude
    # # sanity check
    # plt.plot(np.arange(signal_start, signal_end + 1), signal_data)
    # plt.plot(np.arange(signal_start - int(10 * sampling_rate), signal_start), noise_data)
    # plt.show()
    return snr


def select_traces(distances, sn_ratios, n_bins=200):
    """
    Select a number of traces with the highest signal-to-noise ratios.
    :param distances: distances between the event and stations
    :param sn_ratios: signal-to-noise ratios of the phase
    :return: an array of trace indices
    """
    # divide the whole distance range into n bins
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    dist_bins = np.linspace(np.floor(min_distance), np.ceil(max_distance), n_bins)
    indices_max = []
    dist_bin_counts = []

    for j in range(len(dist_bins) - 1):
        # find indices of traces within the distance range
        indices = np.where((distances >= dist_bins[j]) & (distances < dist_bins[j + 1]))[0]
        dist_bin_counts.append(len(indices))
        if len(indices) != 0:
            sn_ratios_select = sn_ratios[indices]
            # find the trace index with the highest signal-to-noise ratio
            indices_max.append(indices[np.argmax(sn_ratios_select)])
    dist_bin_counts = np.array(dist_bin_counts)
    indices_max = np.array(indices_max)

    return dist_bins, dist_bin_counts, indices_max


def get_travel_time(phase, event_depth, distances=None, min_d=0., max_d=90., epsilon=Constants.epsilon, first=False,
                    all=False):
    """
    This function calculates the predicted travel times for the given phase at given distances/distance range.
    :param phase: str, name of the phase
    :param event_depth: float, event depth
    :param distances: array, distances at which travel times are calculated
    :param min_d: float, minimum distance, default: 0
    :param max_d: float, maximum distance, default: 90
    :param epsilon: float, step size for distance, default: Constants.epsilon
    :param first: boolean, if true, at a given distance, only the time for the first arrival is returned,
    useful for phases with multiple arrivals at a given distance (e.g., P in the triplication region), default: False
    :param all: boolean, if true, return travel times for all distances;
    if the travel time at a given distance is undefined, it is set to NaN;
    otherwise, return travel times at distance ranges only at which they are valid, default: False
    :return:
    distances_new: array, distances
    travel_times: array, travel times at given distances
    """
    # define PREM model
    model = TauPyModel("prem")

    if distances is None:
        distances = np.arange(min_d, max_d + epsilon, epsilon)
    distances_new = []
    travel_times = []
    ray_params = []
    for distance in distances:
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance,
                                          phase_list=[phase])
        if len(arrivals) > 0:
            if first:
                distances_new.append(distance)
                travel_times.append(arrivals[0].time)
                ray_params.append(arrivals[0].ray_param)
            else:
                for arrival in arrivals:
                    distances_new.append(distance)
                    travel_times.append(arrival.time)
                    ray_params.append(arrival.ray_param)
        else:
            if all:
                distances_new.append(distance)
                travel_times.append(np.nan)
                ray_params.append(np.nan)

    return np.array(distances_new), np.array(travel_times), np.array(ray_params)


def get_travel_time_for_plotting(phase, event_depth, distances=None, min_d=0., max_d=90., epsilon=Constants.epsilon):
    """
    This function calculates the predicted travel times for the given phase at given distances/distance range.
    :param phase: str, name of the phase
    :param event_depth: float, event depth
    :param distances: array, distances at which travel times are calculated
    :param min_d: float, minimum distance, default: 0
    :param max_d: float, maximum distance, default: 90
    :param epsilon: float, step size for distance, default: Constants.epsilon
    :return:
    distances_new: array, distances
    travel_times: array, travel times at given distances
    """
    # define PREM model
    model = TauPyModel("prem")

    if distances is None:
        distances = np.arange(min_d, max_d + epsilon, epsilon)
    distances_first = []
    travel_times_first = []
    distances_all = []
    travel_times_all = []
    for distance in distances:
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance,
                                          phase_list=[phase])
        if len(arrivals) > 0:
            distances_first.append(distance)
            travel_times_first.append(arrivals[0].time)
            for arrival in arrivals:
                distances_all.append(distance)
                travel_times_all.append(arrival.time)

    return np.array(distances_first), np.array(travel_times_first), np.array(distances_all), np.array(travel_times_all)


def rotate_to_ned_coordinates(m_uu, m_ss, m_ee, m_us, m_ue, m_se):
    """
    This function takes in 6 moment tensors in Up, South and East coordinates and
    returns the moment tensor matrix in North, East and Down coordinates.
    :param m_uu: float, Up-Up component of the moment tensor
    :param m_ss: float, South-South component of the moment tensor
    :param m_ee: float, East-East component of the moment tensor
    :param m_us: float, Up-South component of the moment tensor
    :param m_ue: float, Up-East component of the moment tensor
    :param m_se: float, South-East component of the moment tensor
    :return: m, array, 3x3 moment tensor matrix in North, East and Down coordinates
    """
    m = np.array([[m_ss, -m_se, m_us],
                  [-m_se, m_ee, -m_ue],
                  [m_us, -m_ue, m_uu]])

    return m
