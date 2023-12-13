import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize

from helper_functions import *


def smooth(data, n_points=5):
    data_len = len(data)
    # calculate the moving average
    data_smooth = np.zeros_like(data)
    for k in range(data_len):
        # edge cases
        if k - n_points < 0:
            data_smooth[k] = np.mean(data[0:k + n_points + 1])
        elif k + n_points + 1 > data_len:
            data_smooth[k] = np.mean(data[k - n_points:data_len])
        else:
            data_smooth[k] = np.mean(data[k - n_points:k + n_points + 1])
    return data_smooth


def plot_traces(A, distances, ax, scale=0.2, picks=None, qualities=None, corr_vals=None):
    n = A.shape[1]
    half_win_len_tmp = int(A.shape[0] / 2)
    for i in range(n):
        a = A[:, i]
        distance = distances[i]
        if corr_vals is not None:
            corr_val = corr_vals[i]
            ax.plot(distance + np.sign(corr_val) * a / max(abs(a)) * scale,
                    np.arange(-half_win_len_tmp, half_win_len_tmp + 1, 1) / sampling_rate, c='k')
        else:
            ax.plot(distance + a / max(abs(a)) * scale,
                    np.arange(-half_win_len_tmp, half_win_len_tmp + 1, 1) / sampling_rate, c='k')
    if (picks is not None) and (qualities is not None):
        ax.scatter(distances, picks, s=100, c=color_map(qualities), zorder=10)
    ax.set_ylim([-half_win_len, half_win_len])
    ax.set_ylabel("Time (s)")


def cross_correlate_low_rank(U, S, Vh, ref_trace_data, ref_trace_window, ref_trace_pick,
                             ref_trace_signal_window_weights, ref_trace_noise_window_weights, p):
    """
    Perform cross-correlation on a low-rank approximation
    :return:
    """
    A_reconstruct = np.zeros((U.shape[0], Vh.shape[1]))
    for k in range(p):
        u = U[:, k][..., np.newaxis]
        vh = Vh[k, :][np.newaxis, ...]
        A_reconstruct += S[k] * u @ vh

    stream_data_reconstruct = A_reconstruct.T
    final_windows = np.zeros((n_traces, 2))
    total_time_shifts = np.zeros(n_traces)
    for i, trace_data_reconstruct in enumerate(stream_data_reconstruct):
        # select a 10 s window centered at the predicted time
        total_time_shift, selected_start, selected_end = select_trace_window(trace_data_reconstruct, sampling_rate,
                                                                             predicted=True)
        final_windows[i] = [selected_start, selected_end]
        total_time_shifts[i] = total_time_shift

    # print(stream_data_reconstruct.shape)
    # print(final_windows.shape)
    # print(ref_trace_data)
    # print(ref_trace_window)

    # cross-correlate with the reference trace
    corr_vals, curr_time_shifts = cross_correlate(stream_data_reconstruct, sampling_rate,
                                                  init_windows=final_windows,
                                                  ref_trace_data=ref_trace_data,
                                                  ref_trace_window=ref_trace_window,
                                                  flip_polarity=flip_polarity)

    # update total time shifts
    total_time_shifts = total_time_shifts + curr_time_shifts

    # calculate pick times
    picks = ref_trace_pick + total_time_shifts / sampling_rate

    # calculate qualities
    sn_ratios = np.zeros(n_traces)
    for i in range(n_traces):
        trace_data_reconstruct = stream_data_reconstruct[i]
        pick = picks[i]
        snr = calculate_snr_by_pick(trace_data_reconstruct, sampling_rate, pick,
                                    signal_window_len=5, noise_window_len=5,
                                    signal_weights=ref_trace_signal_window_weights,
                                    noise_weights=ref_trace_noise_window_weights)  # TODO: add weights
        # update the signal-to-noise ratio
        sn_ratios[i] = snr

    qualities = np.log(sn_ratios) * corr_vals

    return A_reconstruct, corr_vals, total_time_shifts, picks, qualities


def plot_on_map(st_lats, st_lons, picks, qualities_normalized, distance_contours, distance_contour_levels, ax):
    # saturate the color bar
    # lb, ub = -0.4, 1.7  # clean
    # lb, ub = 0.2, 2.3  # crossover
    picks = np.where(picks < lb, lb, picks)
    picks = np.where(picks > ub, ub, picks)
    # normalize
    picks_normalized = (picks - lb) / (ub - lb)

    # sort
    sort_indices = np.argsort(qualities_normalized)

    ax.set_extent([128, 147, 30, 46])
    ax.add_feature(cfeature.LAND, facecolor=[0.5, 0.5, 0.5])
    ax.add_feature(cfeature.OCEAN, facecolor=[0.8, 0.8, 0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([130, 135, 140, 145])
    gl.ylocator = mticker.FixedLocator([32, 36, 40, 44])

    # plot distance contours
    LON, LAT, Z = distance_contours[0], distance_contours[1], distance_contours[2]
    CS = ax.contour(LON, LAT, Z, levels=distance_contour_levels,
                    colors='dimgrey', linewidths=0.5)
    # ax.clabel(CS, CS.levels, manual=manual_locations)

    color_map = cm.get_cmap('seismic')

    # plot picks
    ax.scatter(st_lons[sort_indices], st_lats[sort_indices], s=qualities_normalized[sort_indices] * 50,
               color=color_map(picks_normalized[sort_indices]), alpha=0.8, transform=ccrs.PlateCarree())

    # # add color bar
    # norm = Normalize(lb, ub)
    #
    # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, orientation='vertical', pad=0.02,
    #                   location='right', shrink=0.8)
    # cb.set_label(label="Residual time (s)")


# # clean
# ev_time_str = "20200801170902100000"
# lb, ub = -0.4, 1.7
# crossover_plot = False
# distance_contour_levels = np.arange(25, 42.5, 2.5)
# crossover
ev_time_str = "20110210143927700000"
lb, ub = 0.2, 2.3  # crossover
crossover_plot = True
distance_contour_levels = np.arange(27.5, 47.5, 2.5)
ev_dir = "../../../data/Hinet/deep/" + ev_time_str
phase = 'PcP'
data_dir = "../../data2/" + ev_time_str + "/" + phase
plot_dir_base = "../../plots_other/examples_autopick/" + ev_time_str
plot_dir = plot_dir_base + "/" + phase
save_dir = "../../../../../classes/AM205/final_project/plots"
flip_polarity = False
color_map = cm.get_cmap('viridis')

# load event info
events = pd.read_csv("../../../data/CMT/05_20_PcP_deep_by_quality_path.csv")
ev_time_strs = events['time'].values
ev_idx = np.where(ev_time_strs == ev_time_str)[0][0]
ev = events.iloc[ev_idx]
origin_time = datetime.datetime.strptime(ev['time'], "%Y%m%d%H%M%S%f")
ev_time_JST = UTCDateTime(origin_time) + timedelta(hours=9)
ev_lat = ev['latitude']
ev_lon = ev['longitude']
ev_dep = ev['depth']
ev_mag = ev['mag']
# ev_dir = ev['path']
M = rotate_to_ned_coordinates(ev['m_rr'], ev['m_tt'], ev['m_pp'], ev['m_rt'], ev['m_rp'], ev['m_tp'])

if not os.path.isdir(plot_dir_base):
    os.mkdir(plot_dir_base)

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# load reference trace
with open(data_dir + "/" + ev_time_str + "_reference_trace_metadata.json", 'r') as fp:
    ref_trace_metadata = json.load(fp)

ref_trace_data = ref_trace_metadata['data'][-1]
ref_trace_pick = ref_trace_metadata['pick'][-1]
# select the cross-correlation window and calculate the signal-to-noise of the reference trace
ref_trace_window, ref_trace_pick_new = \
    select_cross_correlation_window(ref_trace_data, sampling_rate, ref_trace_pick)
# calculate the envelope
analytic_signal = hilbert(ref_trace_data)
amplitude_envelope = np.abs(analytic_signal)
# normalize
amplitude_envelope /= max(amplitude_envelope)
# calculate weights for signal and noise
ref_trace_signal_window, _ = \
    select_window(ref_trace_data, sampling_rate, ref_trace_pick,
                  time_before_pick=0, win_len=5)
ref_trace_signal_window_weights = amplitude_envelope[
                                  ref_trace_signal_window[0]:ref_trace_signal_window[1] + 1]
ref_trace_noise_window_weights = np.ones(5 * sampling_rate)

# load station info
stations = pd.read_csv("../../../data/Hinet/hinet_stations.csv")
st_names = np.array([st_name[-4:] for st_name in stations['name'].values])
st_lats = stations['latitude'].values
st_lons = stations['longitude'].values
st_lat_avg = np.mean(st_lats)
st_lon_avg = np.mean(st_lons)

# # calculate the backazimuth
# _, _, baz = calculate_distaz(st_lat_avg, st_lon_avg, ev_lat, ev_lon)
# # convert to degree
# baz = baz / np.pi * 180
#
# bins = [[20, 40], [40, 60], [140, 150], [150, 160], [160, 170], [190, 195], [195, 200], [200, 205], [205, 210],
#         [210, 230], [260, 270], [290, 300]]
# # bins = np.array(bins)
# bin_centers = np.zeros(len(bins))
# for i, bin in enumerate(bins):
#     bin_center = np.mean(bin)
#     bin_centers[i] = bin_center
#
# # load the dc shifts
# with open("../../data_other/dc_shifts_0.2.json", 'r') as fp:
#     dc_shift_dic = json.load(fp)
#
# # find the dc shift of the event
# for i, bin_customize in enumerate(bins):
#     if bin_customize[0] < baz <= bin_customize[1]:
#         bin_center = bin_centers[i]
#         dc_shift = dc_shift_dic[str(bin_center)]

station_names, stream_data, distances, _, _, predicted_windows, _, _, _, _, _, _, _ = \
    preprocess(ev_dir, origin_time, ev_lat, ev_lon, ev_dep, ev_mag, M, phase, data_dir, plot_dir)
n_traces = len(station_names)

# load time shifts
with open(data_dir + "/" + ev_time_str + "_results.json", 'r') as fp:
    results = json.load(fp)

st_lats = np.zeros(n_traces)
st_lons = np.zeros(n_traces)
time_shifts = np.zeros(n_traces)
corr_vals = np.zeros(n_traces)
qualities = np.zeros(n_traces)

for j, st_name in enumerate(station_names):
    st_idx = np.where(st_names == st_name)[0][0]
    st = stations.iloc[st_idx]
    st_lat = st['latitude']
    st_lon = st['longitude']
    st_result = results[st_name]
    time_shift = st_result['time_shift']
    corr_val = st_result['corr']
    quality = st_result['qf']
    st_lats[j] = st_lat
    st_lons[j] = st_lon
    time_shifts[j] = time_shift
    corr_vals[j] = corr_val
    qualities[j] = quality

# svd_windows = predicted_windows + np.vstack([time_shifts, time_shifts]).T
svd_windows = predicted_windows

# form the data matrix
A = [None] * n_traces
for j in range(n_traces):
    trace_data = stream_data[j]
    # # select a 20s window centered at the predicted time
    # _, start, end = select_trace_window(trace_data, sampling_rate, init_window_len=20, predicted=True)
    final_window = svd_windows[j]
    start = final_window[0]
    end = final_window[1]
    corr_val = corr_vals[j]
    # # align by time shift and correct the polarity
    # a = trace_data[int(np.around(start)):int(np.around(end + 1))] * np.sign(corr_val)
    a = trace_data[int(np.around(start)):int(np.around(end + 1))]
    A[j] = a
A = np.transpose(A)

print("Data matrix shape:", A.shape)

# apply SVD
U, S, Vh = np.linalg.svd(A)
# print(U.shape, S.shape, Vh.shape)

plt.rcParams.update({'font.size': 16})
_, ax_sig = plt.subplots(1, 2, figsize=(12, 5), sharex="all")
ax_sig[0].plot(S, 'r')
ax_sig[0].set_ylabel("Singular value ($\sigma$)")
# S_diff = np.diff(S)
S_diff = S[:-1] - S[1:]
# smooth
S_diff_smooth = smooth(S_diff, n_points=5)
ax_sig[1].plot(np.arange(len(S_diff)) + 1, S_diff, 'b')
ax_sig[1].plot(np.arange(len(S_diff_smooth)) + 1, S_diff_smooth, 'darkorange', lw=2, label='10-point moving average')
ax_sig[1].set_ylabel("$\sigma_{i} - \sigma_{i+1}$")
# plt.show()

# original
p = len(S)
# note that A_reconstruct1 is just the original matrix
A_reconstruct1, corr_vals1, total_time_shifts1, picks1, qualities1 = \
    cross_correlate_low_rank(U, S, Vh, ref_trace_data, ref_trace_window, ref_trace_pick_new,
                             ref_trace_signal_window_weights, ref_trace_noise_window_weights, p)

# low-rank approximation
# p = 15
# find the first point where S_diff_smooth < threshold
idx = np.where(S_diff_smooth < 0.2)[0][0]
p = idx + 1
print("cutoff:", p)
A_reconstruct2, corr_vals2, total_time_shifts2, picks2, qualities2 = \
    cross_correlate_low_rank(U, S, Vh, ref_trace_data, ref_trace_window, ref_trace_pick_new,
                             ref_trace_signal_window_weights, ref_trace_noise_window_weights, p)

# indicate on the singular value plot
ax_sig[0].axvline(x=p, c='grey', ls='--')
ax_sig[1].axvline(x=p, c='grey', ls='--', label='cutoff (k)')
ax_sig[1].legend(fontsize=14)
plt.savefig(save_dir + "/" + ev_time_str + "_k_cutoff.png")

# saturate the color bar
min_quality, max_quality = 0, 1
# normalize
min_max_scale_quality = max_quality - min_quality
qualities1 = np.where(qualities1 > max_quality, max_quality, qualities1)
qualities2 = np.where(qualities2 > max_quality, max_quality, qualities2)
qualities1_normalized = (qualities1 - min_quality) / min_max_scale_quality
qualities2_normalized = (qualities2 - min_quality) / min_max_scale_quality

# plot a selected number of traces
_, _, select_indices = select_traces(distances, qualities, n_bins=100)
distances_selected = distances[select_indices]
A_selected = A[:, select_indices]
picks1_selected = picks1[select_indices]
qualities1_normalized_selected = qualities1_normalized[select_indices]
corr_vals1_selected = corr_vals1[select_indices]
A_reconstruct_selected2 = A_reconstruct2[:, select_indices]
picks2_selected = picks2[select_indices]
qualities2_normalized_selected = qualities2_normalized[select_indices]
corr_vals2_selected = corr_vals2[select_indices]

plt.rcParams.update({'font.size': 20})
_, ax = plt.subplots(2, 1, figsize=(20, 7), sharex='all', sharey='all')
sort_indices = np.argsort(qualities1_normalized_selected)  # sort by quality
plot_traces(A_selected[:, sort_indices], distances_selected[sort_indices], ax[0], picks=picks1_selected[sort_indices],
            qualities=qualities1_normalized_selected[sort_indices], corr_vals=corr_vals1_selected[sort_indices])
# # plot the target trend
# ax[0].axhline(y=0, color='hotpink', ls='--', lw=4, alpha=0.8, zorder=11)
# plot the crossover trend
min_distance = np.min(distances)
max_distance = np.max(distances)
# PcP
PcP_distances_first, PcP_travel_times_first, PcP_distances_all, PcP_travel_times_all = \
    get_travel_time_for_plotting('PcP', ev_dep, max_d=max_distance)
if crossover_plot:
    # sP
    sP_distances_first, sP_travel_times_first, sP_distances_all, sP_travel_times_all = \
        get_travel_time_for_plotting('sP', ev_dep, max_d=max_distance)
    # find the common distances
    sP_distances_first_reduced = []
    sP_travel_times_first_reduced = []
    for i, sP_distance in enumerate(sP_distances_first):
        if sP_distance in PcP_distances_first:
            idx_tmp = np.where(PcP_distances_first == sP_distance)[0][0]
            sP_distances_first_reduced.append(sP_distance)
            sP_travel_times_first_reduced.append(sP_travel_times_first[i] - PcP_travel_times_first[idx_tmp])
    ax[0].plot(sP_distances_first_reduced, sP_travel_times_first_reduced, color='magenta', ls='--', lw=4, alpha=0.8,
               zorder=11)
# plot the background picks
sort_indices = np.argsort(qualities1_normalized)  # sort by quality
ax[0].scatter(distances[sort_indices], picks1[sort_indices], s=100, c=color_map(qualities1_normalized[sort_indices]),
              zorder=1, alpha=0.5)
ax[0].set_xlim([min_distance - 1., max_distance + 1.])
# add color bar
norm = Normalize(min_quality, max_quality)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax[0],
                  orientation='vertical', location='right', pad=0.02)
cb.set_label(label="Quality")
# cb.ax.tick_params(labelsize=24)
ax[1].set_xlabel("Distance (degree)")

sort_indices = np.argsort(qualities2_normalized_selected)  # sort by quality
plot_traces(A_reconstruct_selected2[:, sort_indices], distances_selected[sort_indices], ax[1],
            picks=picks2_selected[sort_indices],
            qualities=qualities2_normalized_selected[sort_indices], corr_vals=corr_vals2_selected[sort_indices])
# plot the background picks
sort_indices = np.argsort(qualities2_normalized)  # sort by quality
ax[1].scatter(distances[sort_indices], picks2[sort_indices], s=100, c=color_map(qualities2_normalized[sort_indices]),
              zorder=0, alpha=0.5)
# add color bar
norm = Normalize(min_quality, max_quality)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax[1],
                  orientation='vertical', location='right', pad=0.02)
cb.set_label(label="Quality")
plt.tight_layout()
plt.savefig(save_dir + "/" + ev_time_str + "_waveforms.png")

# # tendency of pick in the middle?
# plt.figure()
# # sort by quality
# sort_indices = np.argsort(qualities1_normalized)
# plt.scatter(total_time_shifts1[sort_indices] / sampling_rate, total_time_shifts2[sort_indices] / sampling_rate, c=color_map(qualities1_normalized[sort_indices]))
# plt.plot([-5, 5], [-5, 5], c='grey', lw=1)
# plt.axis("equal")
# plt.xlabel("Time shift before (s)")
# plt.ylabel("Time shift after (s)")
# # plt.show()
# plt.savefig(save_dir + "/" + ev_time_str + "_scatter_comparison.png")

# plot on map
plt.rcParams.update({'font.size': 14})

# # preprocessing
# plt.figure()
# plt.hist(picks2, bins=np.arange(-10, 10.1, 0.1))
# plt.show()

# create distance contours
LON = np.arange(125, 150.1, 0.1)
LAT = np.arange(30, 46.1, 0.1)
LON, LAT = np.meshgrid(LON, LAT)
Z = np.zeros_like(LON)
for i in range(LON.shape[0]):
    for j in range(LON.shape[1]):
        lon, lat = LON[i][j], LAT[i][j]
        distance, _, _ = calculate_distaz(lat, lon, ev_lat, ev_lon)
        Z[i][j] = distance

# without SVD
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection=ccrs.PlateCarree())  # ccrs stands for cartopy coordinate reference system
plot_on_map(st_lats, st_lons, picks1, qualities1_normalized, (LON, LAT, Z), distance_contour_levels, ax1)
plt.savefig(save_dir + "/" + ev_time_str + "_map_before.png")

# with SVD
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection=ccrs.PlateCarree())  # ccrs stands for cartopy coordinate reference system
plot_on_map(st_lats, st_lons, picks2, qualities2_normalized, (LON, LAT, Z), distance_contour_levels, ax2)
plt.savefig(save_dir + "/" + ev_time_str + "_map_after.png")

# plt.show()
